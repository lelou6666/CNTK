//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include "SequencePacker.h"
#include "ElementTypeUtils.h"

namespace Microsoft { namespace MSR { namespace CNTK {

SequencePacker::SequencePacker(
    MemoryProviderPtr memoryProvider,
    TransformerPtr transformer,
    size_t minibatchSize,
    const std::vector<StreamDescriptionPtr>& streams) : m_transformer(transformer),
    m_minibatchSize(minibatchSize),
    m_outputStreams(streams),
    m_memoryProvider(memoryProvider)
{
    m_inputStreams = m_transformer->GetStreamDescriptions();
    assert(m_inputStreams.size() == m_outputStreams.size());
    assert(
        std::find_if(
        m_outputStreams.begin(),
        m_outputStreams.end(),
        [](const StreamDescriptionPtr& s)
    {
        return s->m_storageType == StorageType::sparse_csc;
    }) == m_outputStreams.end());

    assert(m_minibatchSize > 0);
    for (int i = 0; i < m_outputStreams.size(); ++i)
    {
        const auto& stream = m_outputStreams[i];
        UNUSED(stream);

        // Input and output should match in everything except for sparse/dense.
        assert(stream->m_elementType == ElementType::tfloat || stream->m_elementType == ElementType::tdouble);
        assert(stream->m_name == m_inputStreams[i]->m_name);
        assert(stream->m_id == m_inputStreams[i]->m_id);
        assert(GetSampleSize(m_inputStreams[i]) == GetSampleSize(stream));

        m_streamBufferSizes.push_back(0);
        m_streamBuffers.push_back(nullptr);
    }
}

Minibatch SequencePacker::ReadMinibatch()
{
    auto sequences = m_transformer->GetNextSequences(m_minibatchSize);

    Minibatch minibatch(sequences.m_endOfEpoch);
    if (sequences.m_data.empty())
    {
        return minibatch;
    }

    minibatch.m_data.reserve(sequences.m_data.size());
    for (size_t streamIndex = 0; streamIndex < sequences.m_data.size(); ++streamIndex)
    {
        minibatch.m_data.push_back(PackStreamMinibatch(sequences.m_data[streamIndex], streamIndex));
    }

    return minibatch;
}

StreamMinibatchPtr SequencePacker::PackStreamMinibatch(const std::vector<SequenceDataPtr>& sequences, size_t streamId)
{
    std::vector<MBLayout::SequenceInfo> inputSequences;
    for (size_t index = 0; index < sequences.size(); ++index)
    {
        MBLayout::SequenceInfo info;

        // In each minibatch sequence ids should be unique.
        // They have to match between different input streams in the same minibatch.
        // We are using sequence index in the set of received sequences.
        // TODO: should we use m_key as sequence id and pass it with data?
        info.seqId = index;

        info.tBegin = 0;
        info.tEnd = sequences[index]->m_numberOfSamples;
        inputSequences.push_back(info);
    }

    std::vector<std::pair<size_t, size_t>> placement;
    std::vector<size_t> rowAllocations;

    // Creating the minibatch layout.
    MBLayoutPtr layout = std::make_shared<MBLayout>();
    layout->InitAsPackedSequences(inputSequences, placement, rowAllocations);

    // Allocating necessary buffer for the stream.
    size_t sampleSize = GetSampleSize(m_inputStreams[streamId]);
    size_t totalNumberOfSamples = layout->GetNumCols() * sampleSize;
    if (m_streamBufferSizes[streamId] < totalNumberOfSamples)
    {
        m_streamBuffers[streamId] = AllocateBuffer(layout->GetNumCols(), sampleSize);
        m_streamBufferSizes[streamId] = totalNumberOfSamples;
    }

    // Identify a stride for two adjecent sample of the same sequence.
    // Note, sequences are packed coalesced
    // (s11, s21, ... sN1 (here stride is finished) | s12, s22, ... sN2 (here stride is finished) | ...)
    // to make sure efficient execution on GPU.
    size_t stride = GetSampleSize(m_inputStreams[streamId]) * layout->GetNumParallelSequences();

    // Packing the actual data.
    StorageType storageType = m_inputStreams[streamId]->m_storageType;
    size_t elementSize = GetSizeByType(m_inputStreams[streamId]->m_elementType);
    const auto& packedSequences = layout->GetAllSequences();
    for (const auto& sequence : packedSequences)
    {
        if (sequence.seqId == GAP_SEQUENCE_ID)
            continue;
        const auto& data = sequences[sequence.seqId];

        // Packing the sequence
        // The resulting sequence should currently be dense!
        for (size_t sampleIndex = 0; sampleIndex < sequence.GetNumTimeSteps(); ++sampleIndex)
        {
            char* destination = m_streamBuffers[streamId].get() + layout->GetColumnIndex(sequence, sampleIndex) * stride + sequence.s * sampleSize;
            if (storageType == StorageType::dense)
            {
                PackDenseSample(destination, data, sampleIndex, elementSize, sampleSize);
            }
            else // sparse
            {
                PackSparseSample(destination, data, sampleIndex, elementSize, sampleSize);
            }
        }
    }

    // Ok, minibatch is ready, give it out.
    StreamMinibatchPtr result = std::make_shared<StreamMinibatch>();
    result->m_data = m_streamBuffers[streamId].get();
    result->m_dataSize = m_streamBufferSizes[streamId] * GetSampleSize(m_outputStreams[streamId]);
    result->m_layout = layout;
    return result;
}

void SequencePacker::PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize)
{
    // Setting buffer to 0.
    memset(destination, 0, sampleSize);

    SparseSequenceDataPtr s = static_pointer_cast<SparseSequenceData>(sequence);
    size_t nonZeroCount = s->m_indices[sample].size();
    for (size_t nonZeroIndex = 0; nonZeroIndex < nonZeroCount; ++nonZeroIndex)
    {
        memcpy(
            (char*)destination + s->m_indices[sample][nonZeroIndex] * elementSize,
            (const char*)(s->m_data) + nonZeroIndex * elementSize,
            elementSize);
    }
}

void SequencePacker::PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t /*elementSize*/, size_t sampleSize)
{
    memcpy(destination, (char*)(sequence->m_data) + sample * sampleSize, sampleSize);
}

size_t SequencePacker::GetSampleSize(StreamDescriptionPtr stream)
{
    assert(stream != nullptr);
    size_t elementSize = GetSizeByType(stream->m_elementType);
    return stream->m_sampleLayout->GetNumElements() * elementSize;
}

std::shared_ptr<char> SequencePacker::AllocateBuffer(size_t numElements, size_t elementSize)
{
    return std::shared_ptr<char>(
        reinterpret_cast<char*>(m_memoryProvider->Alloc(elementSize, numElements)),
        [this](char* p)
    {
        m_memoryProvider->Free(p);
    });
}

}}}
