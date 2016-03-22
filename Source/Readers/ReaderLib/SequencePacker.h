//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Reader.h"
#include "MemoryProvider.h"
#include "Transformer.h"
#include "Packer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A sequence packer that packs dense or sparse samples in dense minibatch for parallel GPU consumption.
class SequencePacker : public Packer
{
public:
    SequencePacker(
        MemoryProviderPtr memoryProvider,
        TransformerPtr transformer,
        size_t minibatchSize,
        const std::vector<StreamDescriptionPtr>& streams);

    virtual Minibatch ReadMinibatch() override;

private:
    // Auxiliary packing functions.
    StreamMinibatchPtr PackStreamMinibatch(const std::vector<SequenceDataPtr>& sequences, size_t streamId);
    void PackSparseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);
    void PackDenseSample(void* destination, SequenceDataPtr sequence, size_t sample, size_t elementSize, size_t sampleSize);

    // Utility functions.
    std::shared_ptr<char> AllocateBuffer(size_t numElements, size_t elementSize);
    size_t GetSampleSize(StreamDescriptionPtr stream);

    MemoryProviderPtr m_memoryProvider;
    TransformerPtr m_transformer;
    std::vector<StreamDescriptionPtr> m_outputStreams;
    std::vector<StreamDescriptionPtr> m_inputStreams;
    size_t m_minibatchSize;

    // Buffers for allocated data.
    std::vector<std::shared_ptr<char>> m_streamBuffers;
    std::vector<size_t> m_streamBufferSizes;
};

typedef std::shared_ptr<SequencePacker> SequencePackerPtr;

}}}
