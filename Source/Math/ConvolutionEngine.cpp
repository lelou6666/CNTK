//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ConvolutionEngine.h"
#include "CuDnnConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void ConvolutionEngine<ElemType>::Forward(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, 
                                          const ConvDesc& convDesc, const Tensor4D& outT, Mat& out, Mat& workspace)
{
    assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
    assert(inT.n() == in.GetNumCols());
    assert(filterT.k() == filter.GetNumRows());
    assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());
    assert(inT.c() == filterT.c());
    assert(outT.c() == filterT.k());
    assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
    assert(outT.n() == out.GetNumCols());

    EnsureCompatible();
    ForwardCore(inT, in, filterT, filter, convDesc, outT, out, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardData(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
                                               const Tensor4D& gradT, Mat& grad, Mat& workspace)
{
    assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
    assert(srcGradT.n() == srcGrad.GetNumCols());
    assert(filterT.k() == filter.GetNumRows());
    assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());
    assert(srcGradT.c() == filterT.k());
    assert(gradT.c() == filterT.c());
    assert(gradT.w() * gradT.h() * gradT.c() == grad.GetNumRows());
    assert(gradT.n() == grad.GetNumCols());

    EnsureCompatible();
    BackwardDataCore(srcGradT, srcGrad, filterT, filter, convDesc, gradT, grad, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardFilter(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc,
                                                 const Filter& filterT, Mat& filter, bool allowReuse, Mat& workspace)
{
    assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
    assert(srcGradT.n() == srcGrad.GetNumCols());
    assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
    assert(inT.n() == in.GetNumCols());
    assert(srcGradT.c() == filterT.k());
    assert(inT.c() == filterT.c());
    assert(filterT.k() == filter.GetNumRows());
    assert(filterT.w() * filterT.h() * filterT.c() == filter.GetNumCols());

    EnsureCompatible();
    BackwardFilterCore(srcGradT, srcGrad, inT, in, convDesc, filterT, filter, allowReuse, workspace);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::NormalizeBatch(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                                                 bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out,
                                                 double epsilon, Mat& saveMean, Mat& saveInvStdDev)
{
    const size_t crowIn = inT.w() * inT.h() * inT.c();
    if (spatial)
    {
        assert(scaleBiasT.c() == inT.c());
        assert(scaleBiasT.w() == 1);
        assert(scaleBiasT.h() == 1);
        assert(runMean.GetNumRows() == inT.c());
        assert(runInvStdDev.GetNumRows() == inT.c());
    }
    else
    {
        assert(scaleBiasT.c() == inT.c());
        assert(scaleBiasT.w() == inT.w());
        assert(scaleBiasT.h() == inT.h());
        assert(runMean.GetNumRows() == crowIn);
        assert(runInvStdDev.GetNumRows() == crowIn);
    }
    assert(scaleBiasT.n() == 1);
    assert(crowIn == in.GetNumRows());
    assert(crowIn == out.GetNumRows());
    assert(inT.n() == in.GetNumCols());
    assert(inT.n() == out.GetNumCols());
    assert(bias.GetNumCols() == 1);
    assert(scale.GetNumCols() == 1);
    assert(runMean.GetNumCols() == 1);
    assert(runInvStdDev.GetNumCols() == 1);
    assert(runMean.GetNumCols() == saveMean.GetNumCols());
    assert(runMean.GetNumRows() == saveMean.GetNumRows());
    assert(runInvStdDev.GetNumCols() == saveInvStdDev.GetNumCols());
    assert(runInvStdDev.GetNumRows() == saveInvStdDev.GetNumRows());

#ifndef _DEBUG
    UNUSED(crowIn); // crowIn used only in asserts.
#endif

    EnsureCompatibleBatchNorm(spatial);
    NormalizeBatchCore(inT, in, scaleBiasT, scale, bias, spatial, expAvgFactor, runMean, runInvStdDev, out, epsilon, saveMean, saveInvStdDev);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::NormalizeBatchInference(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                                                          bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out)
{
    const size_t crowIn = inT.w() * inT.h() * inT.c();

    if (spatial)
    {
        assert(scaleBiasT.c() == inT.c());
        assert(scaleBiasT.w() == 1);
        assert(scaleBiasT.h() == 1);
        assert(scaleBiasT.c() == runMean.GetNumRows());
        assert(scaleBiasT.c() == runInvStdDev.GetNumRows());
    }
    else
    {
        assert(scaleBiasT.c() == inT.c());
        assert(scaleBiasT.w() == inT.w());
        assert(scaleBiasT.h() == inT.h());
        assert(crowIn == runMean.GetNumRows());
        assert(crowIn == runInvStdDev.GetNumRows());
    }
    assert(scaleBiasT.n() == 1);
    assert(crowIn == in.GetNumRows());
    assert(crowIn == out.GetNumRows());
    assert(inT.n() == in.GetNumCols());
    assert(inT.n() == out.GetNumCols());
    assert(bias.GetNumCols() == 1);
    assert(scale.GetNumCols() == 1);
    assert(runMean.GetNumCols() == 1);
    assert(runInvStdDev.GetNumCols() == 1);
#ifndef _DEBUG
    // used only in asserts.
    UNUSED(crowIn);
#endif

    EnsureCompatibleBatchNorm(spatial);
    NormalizeBatchInferenceCore(inT, in, scaleBiasT, scale, bias, spatial, runMean, runInvStdDev, out);
}

template <class ElemType>
void ConvolutionEngine<ElemType>::BackwardNormalizeBatch(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad,
                                                         const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
                                                         Mat& scaleGrad, Mat& biasGrad)
{
    const size_t crowIn = inT.w() * inT.h() * inT.c();

    if (spatial)
    {
        assert(scaleBiasT.c() == inT.c());
        assert(scaleBiasT.w() == 1);
        assert(scaleBiasT.h() == 1);
    }
    else
    {
        assert(scaleBiasT.c() == inT.c());
        assert(scaleBiasT.w() == inT.w());
        assert(scaleBiasT.h() == inT.h());
    }
    assert(scaleBiasT.n() == 1);
    assert(crowIn == in.GetNumRows());
    assert(crowIn == srcGrad.GetNumRows());
    assert(crowIn == grad.GetNumRows());
    assert(inT.n() == in.GetNumCols());
    assert(inT.n() == srcGrad.GetNumCols());
    assert(inT.n() == grad.GetNumCols());
    assert(scaleGrad.GetNumRows() == scale.GetNumRows());
    assert(scaleGrad.GetNumCols() == scale.GetNumCols());
    assert(biasGrad.GetNumRows() == scale.GetNumRows());
    assert(biasGrad.GetNumCols() == scale.GetNumCols());
#ifndef _DEBUG
    UNUSED(crowIn); // crowIn used only in asserts.
#endif

    EnsureCompatibleBatchNorm(spatial);
    BackwardNormalizeBatchCore(inT, in, srcGrad, grad, scaleBiasT, scale, spatial, saveMean, saveInvStdDev, scaleGrad, biasGrad);
}

//------------------------------------------------------------------
// Default (legacy) convolution engine implementation.
//------------------------------------------------------------------
template <class ElemType>
class DefaultConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;
    using typename Base::Tensor4D;
    using typename Base::Filter;
    using typename Base::ConvDesc;

public:
    DefaultConvolutionEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, BatchNormImpl bnImpl)
        : Base(deviceId, imageLayout), m_ones(deviceId), m_maxTempMemSizeInSamples(maxTempMemSizeInSamples), m_bnImpl(bnImpl)
    {
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::HWC)
            RuntimeError("Default convolution engine currently supports only HWC/legacy layout.");
    }

    void ForwardCore(const Tensor4D& inT, const Mat& in, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
                 const Tensor4D& outT, Mat& out, Mat& workspace) override
    {
        size_t packedInputRows = filterT.w() * filterT.h() * filterT.c();
        size_t packedInputColsPerSample = outT.w() * outT.h();
        size_t outputSizePerChannel = packedInputColsPerSample;
        // size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
        // size_t inputDim = inT.w() * inT.h() * inT.c();  // size of each input sample

        size_t batchSize = inT.n();
        size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

        assert(filter.GetNumCols() == packedInputRows && filter.GetNumRows() == outT.c());
        UNUSED(packedInputRows);

        // GPU and 1-dimensional image
        m_gpuSparseOpt = (filterT.h() == 1 &&
                          in.GetCurrentMatrixLocation() == CurrentDataLocation::GPU &&
                          convDesc.wStride() == 1 &&
                          !convDesc.padding() &&
                          in.GetMatrixType() == MatrixType::SPARSE);
        m_gpuSparse1D = (m_gpuSparseOpt && inT.h() == 1);

        out.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, false);

        // Reshaping is only necessary if we are going to use the unpacking trick
        if (m_gpuSparseOpt)
            out.Reshape(outT.c() * outT.w(), outT.h() * batchSize);
        else
            out.Reshape(outT.c(), outputSizePerChannel * batchSize);

        size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
        size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

        for (size_t i = 0; i < numSubBatches; i++)
        {
            size_t startSampleId = i * subBatchSize;
            size_t endSampleId = min(batchSize, startSampleId + subBatchSize);
            size_t smallBatchSize = endSampleId - startSampleId;
            Mat inputSubBatch(in.GetDeviceId());

            // We optimize for three different scenarios here by handling them slightly differently.
            // [Scenario 1] Dense: Unroll using AssignPackedConvolutionInput and multiply.
            // [Scenario 2] Sparse 1-D convolution on GPU: for text scenarios we have a specific kernel.
            // [Scenario 3] Sparse all others: convert to dense. Temporary work-around - allocating/de-allocating memory is costly!
            if (in.GetMatrixType() == MatrixType::DENSE || m_gpuSparse1D)
                inputSubBatch = in.ColumnSlice(startSampleId, smallBatchSize);
            else
                inputSubBatch.SetValue(in.ColumnSlice(startSampleId, smallBatchSize), in.GetFormat());

            if (m_gpuSparseOpt)
            {
                if (filterT.w() * inT.c() != filter.GetNumCols())
                    LogicError("Kernel width and weight matrix dimensions don't match.");

                inputSubBatch.Reshape(inT.c() * inT.w(), inT.h() * smallBatchSize);
                Mat outputSubBatch = out.ColumnSlice(startSampleId, outT.h() * smallBatchSize);
                Mat::ConvolveAndWeightedAdd(1, filter, false, inputSubBatch, false, 0, outputSubBatch,
                                            static_cast<int>(inT.c()), convDesc.wStride(), convDesc.padding(), true);
            }
            else
            {
                inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
                workspace.AssignPackedConvolutionInput(inputSubBatch,
                                                       inT.w(), inT.h(), inT.c(),
                                                       outT.w(), outT.h(), outT.c(),
                                                       filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                                                       convDesc.padding());

                Mat outputSubBatch = out.ColumnSlice(outputSizePerChannel * startSampleId, outputSizePerChannel * smallBatchSize);

                // workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                // BUGBUG: This ^^ destroys the content of the matrix. Also it seems not to change the size. Does it? Should this be a Reshape()?
                Mat::Multiply(filter, false, workspace, false, outputSubBatch);
            }
        }

        out.Reshape(outT.c() * outputSizePerChannel, batchSize); // each sample becomes a column

        assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
        assert(outT.n() == out.GetNumCols());
    }

    void BackwardDataCore(const Tensor4D& srcGradT, const Mat& srcGrad, const Filter& filterT, const Mat& filter, const ConvDesc& convDesc,
                      const Tensor4D& gradT, Mat& grad, Mat& workspace) override
    {
        size_t packedInputRows = filterT.w() * filterT.h() * filterT.c();
        size_t packedInputColsPerSample = srcGradT.w() * srcGradT.h();
        size_t outputSizePerChannel = packedInputColsPerSample;
        // size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
        // size_t inputDim = gradT.w() * gradT.h() * gradT.c();  // size of each input sample

        size_t batchSize = srcGradT.n();

        size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

        // Create slice which is the same as full matrix so we can reshape it.
        Matrix<ElemType> srcGradTmp = srcGrad.ColumnSlice(0, srcGrad.GetNumCols());
        srcGradTmp.Reshape(srcGradT.c(), outputSizePerChannel * batchSize); // reshape to match the longernal operation

        size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
        size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

        for (size_t i = 0; i < numSubBatches; i++)
        {
            size_t startSampleId = i * subBatchSize;
            size_t endSampleId = min(batchSize, startSampleId + subBatchSize);
            size_t smallBatchSize = endSampleId - startSampleId;

            workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
            Matrix<ElemType> outputGradientSubBatch = srcGradTmp.ColumnSlice(startSampleId * outputSizePerChannel, smallBatchSize * outputSizePerChannel);
            Matrix<ElemType>::Multiply(filter, true, outputGradientSubBatch, false, workspace);

            Matrix<ElemType> inputGradientSubBatch = grad.ColumnSlice(startSampleId, smallBatchSize);
            workspace.UnpackConvolutionInput(inputGradientSubBatch,
                                             gradT.w(), gradT.h(), gradT.c(),
                                             srcGradT.w(), srcGradT.h(), srcGradT.c(),
                                             filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                                             convDesc.padding());
        }

        assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
        assert(srcGradT.n() == srcGrad.GetNumCols());
    }

    void BackwardFilterCore(const Tensor4D& srcGradT, const Mat& srcGrad, const Tensor4D& inT, const Mat& in, const ConvDesc& convDesc,
                        const Filter& filterT, Mat& filter, bool allowReuse, Mat& workspace) override
    {
        size_t packedInputRows = filterT.w() * filterT.h() * filterT.c();
        size_t packedInputColsPerSample = srcGradT.w() * srcGradT.h();
        size_t outputSizePerChannel = packedInputColsPerSample;
        // size_t packedInputDim = packedInputRows * packedInputColsPerSample; // size of each packed input sample
        // size_t inputDim = m_inputImageLayout.width * m_inputImageLayout.height * m_inputImageLayout.channels;  // size of each input sample

        size_t batchSize = inT.n();

        size_t maxTempMemSizeInSamples = (m_maxTempMemSizeInSamples == 0 ? batchSize : m_maxTempMemSizeInSamples);

        // const Matrix<ElemType> & weightMatrix = input0;
        // inputGradientValues.Resize(weightMatrix.GetNumRows(), weightMatrix.GetNumCols()); // should have been resized when preparing gradient computation

        // Create slice which is the same as full matrix so we can reshape it.
        Matrix<ElemType> srcGradTmp = srcGrad.ColumnSlice(0, srcGrad.GetNumCols());
        srcGradTmp.Reshape(srcGradT.c(), outputSizePerChannel * batchSize); // reshape to match the longernal operation

        size_t subBatchSize = min(batchSize, maxTempMemSizeInSamples);
        size_t numSubBatches = (batchSize + subBatchSize - 1) / subBatchSize;

        if (numSubBatches == 1 && allowReuse && !m_gpuSparseOpt) // reuse packed input from evaluation step if it's not changed by either subbatch or recurrent steps.
            // REVIEW alexeyk: the following makes an assumption that data in workspace was filled by Forward call and remained unchanged. Find way to enforce/verify that.
            Matrix<ElemType>::MultiplyAndAdd(srcGradTmp, false, workspace, true, filter);
        else
        {
            for (size_t i = 0; i < numSubBatches; i++)
            {
                size_t startSampleID = i * subBatchSize;
                size_t endSampleID = min(batchSize, startSampleID + subBatchSize);
                size_t smallBatchSize = endSampleID - startSampleID;
                Matrix<ElemType> outputGradientSubBatch = srcGradTmp.ColumnSlice(startSampleID * outputSizePerChannel, smallBatchSize * outputSizePerChannel);

                // We optimize for three different scenarios here by handling them slightly differently.
                // [Scenario 1] Dense: Unroll using AssignPackedConvolutionInput and multiply.
                // [Scenario 2] Sparse 1-D convolution on GPU: for text scenarios we have a specific kernel.
                // [Scenario 3] Sparse all others: convert to dense. Temporary work-around - allocating/de-allocating memory is costly!
                if (m_gpuSparseOpt)
                {
                    Matrix<ElemType> inputSubBatch(in.GetDeviceId());
                    inputSubBatch.SetValue(in.ColumnSlice(startSampleID, smallBatchSize));
                    inputSubBatch.Reshape(inT.c(), smallBatchSize * inT.w() * inT.h());
                    Matrix<ElemType> inputSubBatchSparseReordered(inputSubBatch.GetNumCols(), inputSubBatch.GetNumRows(), inputSubBatch.GetDeviceId(), MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC);
                    Matrix<ElemType>::TensorShuffleScaleAndAdd(0.0f, inputSubBatch.Transpose(), 1, inT.w(), 1, smallBatchSize * inT.h(), inT.c(), 1.0f, inputSubBatchSparseReordered, inputSubBatchSparseReordered);

                    Matrix<ElemType> outputGradientSubBatchReordered = Matrix<ElemType>::Zeros(smallBatchSize * srcGradT.h() * srcGradT.w(), srcGradT.c(), outputGradientSubBatch.GetDeviceId());
                    Matrix<ElemType>::TensorShuffleScaleAndAdd(0.0f, outputGradientSubBatch.Transpose(), 1, srcGradT.w(), 1, smallBatchSize * srcGradT.h(), srcGradT.c(), 1.0f, outputGradientSubBatchReordered, outputGradientSubBatchReordered);

                    filter.Reshape(srcGradT.c() * filterT.w(), inT.c());
                    Matrix<ElemType>::ConvolveAndWeightedAdd(1, outputGradientSubBatchReordered, true, inputSubBatchSparseReordered, false, 1, filter, smallBatchSize * inT.h(), convDesc.wStride(), convDesc.padding(), false);
                    filter.Reshape(srcGradT.c(), inT.c() * filterT.w());
                }
                else
                {
                    workspace.Resize(packedInputRows, packedInputColsPerSample * smallBatchSize);
                    Matrix<ElemType> inputSubBatch = in.ColumnSlice(startSampleID, smallBatchSize);
                    inputSubBatch.SwitchToMatrixType(MatrixType::DENSE, inputSubBatch.GetFormat(), true);
                    workspace.AssignPackedConvolutionInput(inputSubBatch,
                                                           inT.w(), inT.h(), inT.c(),
                                                           srcGradT.w(), srcGradT.h(), srcGradT.c(),
                                                           filterT.w(), filterT.h(), convDesc.wStride(), convDesc.hStride(),
                                                           convDesc.padding());

                    Matrix<ElemType>::MultiplyAndAdd(outputGradientSubBatch, false, workspace, true, filter);
                }
            }
        }

        assert(srcGradT.w() * srcGradT.h() * srcGradT.c() == srcGrad.GetNumRows());
        assert(srcGradT.n() == srcGrad.GetNumCols());
    }

    void EnsureCompatibleBatchNorm(bool spatial) override
    {
        if (m_deviceId >= 0)
            InvalidArgument("This engine does not support batch normalization on GPUs.");
        if (m_bnImpl != BatchNormImpl::Cntk)
            InvalidArgument("Only CNTK batch normalization implementation is supported by this engine.");
        if (spatial && m_imageLayout != ImageLayoutKind::CHW)
            InvalidArgument("This engine batch normalization currently supports only CHW data layout for convolutional nodes.");
    }

    void NormalizeBatchCore(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                        bool spatial, double expAvgFactor, Mat& runMean, Mat& runInvStdDev, Mat& out, double epsilon, Mat& saveMean, Mat& saveInvStdDev) override
    {
        UNUSED(inT);
        UNUSED(in);
        UNUSED(scaleBiasT);
        UNUSED(scale);
        UNUSED(bias);
        UNUSED(out);
        UNUSED(spatial);
        UNUSED(expAvgFactor);
        UNUSED(runMean);
        UNUSED(runInvStdDev);
        UNUSED(epsilon);
        UNUSED(saveMean);
        UNUSED(saveInvStdDev);
        RuntimeError("Not yet implemented.");
    }

    void NormalizeBatchInferenceCore(const Tensor4D& inT, const Mat& in, const Tensor4D& scaleBiasT, const Mat& scale, const Mat& bias,
                                 bool spatial, const Mat& runMean, const Mat& runInvStdDev, Mat& out) override
    {
        UNUSED(scaleBiasT);
        if (spatial)
        {
            size_t spatialSize = inT.w() * inT.h();
#pragma omp parallel for
            for (long icol = 0; icol < out.GetNumCols(); icol++)
            {
                for (long irow = 0; irow < out.GetNumRows(); irow++)
                {
                    size_t imap = irow / spatialSize;
                    out(irow, icol) = scale(imap, 0) * (in(irow, icol) - runMean(imap, 0)) * runInvStdDev(imap, 0) + bias(imap, 0);
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (long icol = 0; icol < out.GetNumCols(); icol++)
            {
                for (long irow = 0; irow < out.GetNumRows(); irow++)
                {
                    out(irow, icol) = scale(irow, 0) * (in(irow, icol) - runMean(irow, 0)) * runInvStdDev(irow, 0) + bias(irow, 0);
                }
            }
        }
    }

    void BackwardNormalizeBatchCore(const Tensor4D& inT, const Mat& in, const Mat& srcGrad, Mat& grad,
                                const Tensor4D& scaleBiasT, const Mat& scale, bool spatial, const Mat& saveMean, const Mat& saveInvStdDev,
                                Mat& scaleGrad, Mat& biasGrad) override
    {
        UNUSED(inT);
        UNUSED(in);
        UNUSED(srcGrad);
        UNUSED(grad);
        UNUSED(scaleBiasT);
        UNUSED(scale);
        UNUSED(scaleGrad);
        UNUSED(biasGrad);
        UNUSED(spatial);
        UNUSED(saveMean);
        UNUSED(saveInvStdDev);
        RuntimeError("Not yet implemented.");
    }

private:
    size_t m_maxTempMemSizeInSamples;
    BatchNormImpl m_bnImpl;
    Mat m_ones;
    bool m_gpuSparseOpt;
    bool m_gpuSparse1D;
};

template class ConvolutionEngine<float>;
template class ConvolutionEngine<double>;

//------------------------------------------------------------------
// Pooling engine.
//------------------------------------------------------------------


template <class ElemType>
void PoolingEngine<ElemType>::Forward(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out)
{
    assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
    assert(inT.n() == in.GetNumCols());
    assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
    assert(outT.n() == out.GetNumCols());

    EnsureCompatible();
    ForwardCore(inT, in, poolDesc, outT, out);
}

template <class ElemType>
void PoolingEngine<ElemType>::Backward(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad)
{
    assert(outT.w() * outT.h() * outT.c() == out.GetNumRows());
    assert(outT.n() == out.GetNumCols());
    assert(out.GetNumRows() == srcGrad.GetNumRows());
    assert(out.GetNumCols() == srcGrad.GetNumCols());
    assert(inT.w() * inT.h() * inT.c() == in.GetNumRows());
    assert(inT.n() == in.GetNumCols());
    assert(in.GetNumRows() == grad.GetNumRows());
    assert(in.GetNumCols() == grad.GetNumCols());

    EnsureCompatible();
    BackwardCore(outT, out, srcGrad, poolDesc, inT, in, grad);
}

//------------------------------------------------------------------
// Default (legacy) pooling engine implementation.
//------------------------------------------------------------------
template <class ElemType>
class DefaultPoolingEngine : public PoolingEngine<ElemType>
{
public:
    using Base = PoolingEngine<ElemType>;
    using typename Base::Tensor4D;
    using typename Base::PoolDesc;
    using typename Base::Mat;

public:
    DefaultPoolingEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout)
        : Base(deviceId, imageLayout)
    {
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::HWC)
            RuntimeError("Default pooling engine currently supports only HWC/legacy layout.");
    }

    void ForwardCore(const Tensor4D& inT, const Mat& in, const PoolDesc& poolDesc, const Tensor4D& outT, Mat& out) override
    {
        if (poolDesc.kind() == PoolDesc::PoolKind::Max)
        {
            out.AssignMaxPoolingResult(in, inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                                       outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                                       poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
        }
        else if (poolDesc.kind() == PoolDesc::PoolKind::Average)
        {
            out.AssignAveragePoolingResult(in, inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                                           outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                                           poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)poolDesc.kind());
    }

    void BackwardCore(const Tensor4D& outT, const Mat& out, const Mat& srcGrad, const PoolDesc& poolDesc, const Tensor4D& inT, const Mat& in, Mat& grad) override
    {
        if (poolDesc.kind() == PoolDesc::PoolKind::Max)
        {
            grad.AddMaxPoolingGradient(srcGrad, in, out,
                                       inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                                       outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                                       poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
        }
        else if (poolDesc.kind() == PoolDesc::PoolKind::Average)
        {
            grad.AddAveragePoolingGradient(srcGrad, inT.c(), inT.w(), inT.h(), inT.w() * inT.h() * inT.c(),
                                           outT.w(), outT.h(), outT.w() * outT.h() * outT.c(),
                                           poolDesc.w(), poolDesc.h(), poolDesc.wStride(), poolDesc.hStride());
        }
        else
            InvalidArgument("Pooling type %d is not supported.", (int)poolDesc.kind());
    }
};

template class PoolingEngine<float>;
template class PoolingEngine<double>;

template <class ElemType>
class DefaultConvolutionEngineFactory : public ConvolutionEngineFactory<ElemType>
{
public:
    using Base = ConvolutionEngineFactory<ElemType>;
    using typename Base::Tensor4D;
    using typename Base::Tensor4DPtr;
    using typename Base::Filter;
    using typename Base::FilterPtr;
    using typename Base::ConvDesc;
    using typename Base::ConvDescPtr;
    using typename Base::PoolDesc;
    using typename Base::PoolDescPtr;

    using typename Base::ConvEnginePtr;
    using typename Base::PoolEnginePtr;

public:
    Tensor4DPtr CreateTensor(size_t w, size_t h, size_t c, size_t n) override
    {
        return std::make_unique<ConvolutionTensor4D>(w, h, c, n);
    }

    FilterPtr CreateFilter(size_t w, size_t h, size_t c, size_t k) override
    {
        return std::make_unique<Filter>(w, h, c, k);
    }

    ConvDescPtr CreateConvDescriptor(const Tensor4D& /*inT*/, const Filter& /*filterT*/,
                                     size_t wStride, size_t hStride, bool padding) override
    {
        return std::make_unique<ConvDesc>(wStride, hStride, padding);
    }

    PoolDescPtr CreatePoolDescriptor(typename PoolDesc::PoolKind kind, size_t w, size_t h, size_t wStride, size_t hStride, size_t wPad, size_t hPad) override
    {
        return std::make_unique<PoolDesc>(kind, w, h, wStride, hStride, wPad, hPad);
    }

    ConvEnginePtr CreateConvEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples, BatchNormImpl bnImpl) override
    {
        return std::make_unique<DefaultConvolutionEngine<ElemType>>(deviceId, imageLayout, maxTempMemSizeInSamples, bnImpl);
    }

    PoolEnginePtr CreatePoolEngine(DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout) override
    {
        return std::make_unique<DefaultPoolingEngine<ElemType>>(deviceId, imageLayout);
    }
};

template <class ElemType>
std::unique_ptr<ConvolutionEngineFactory<ElemType>> ConvolutionEngineFactory<ElemType>::Create(DEVICEID_TYPE deviceId, EngineType engType, ImageLayoutKind imageLayoutKind)
{
    if (engType == EngineType::Auto)
    {
        // REVIEW alexeyk: make cuDNN default when running on GPU and compiled with cuDNN, add config parameter to enable runtime switch between implementations.
        if (deviceId >= 0 && CuDnnConvolutionEngineFactory<ElemType>::IsSupported(deviceId) && imageLayoutKind == ImageLayoutKind::CHW)
            return Create(deviceId, EngineType::CuDnn, imageLayoutKind);
        else
            return Create(deviceId, EngineType::Legacy, imageLayoutKind);
    }
    else if (engType == EngineType::CuDnn)
    {
        if (imageLayoutKind != ImageLayoutKind::CHW)
            InvalidArgument("ConvolutionEngineFactory: ImageLayout '%s' is not compatible with the cuDNN engine.", ToString(imageLayoutKind).c_str());
        if (deviceId >= 0 && CuDnnConvolutionEngineFactory<ElemType>::IsSupported(deviceId))
            return std::make_unique<CuDnnConvolutionEngineFactory<ElemType>>();
        RuntimeError("cuDNN convolution engine is not supported, check the device id and whether the code was compiled with cuDNN.");
    }
    else if (engType == EngineType::Legacy)
    {
        return std::make_unique<DefaultConvolutionEngineFactory<ElemType>>();
    }

    RuntimeError("Not supported convolution engine type: %d.", (int)engType);
}

template class ConvolutionEngineFactory<float>;
template class ConvolutionEngineFactory<double>;

}}}
