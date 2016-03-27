//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/Helpers.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(GPUMatrixSuite)

#if 0
// TODO
// These tests currently don't reliably run and need rework.
// When re-enabling please make sure they run reliably by testing
// in several runs and on multiple machines.

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_DefaultBehaviorTestForConstructors, RandomSeedFixture)
{
    const SingleMatrix matrixA(c_deviceIdZero);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(0, matrixA.GetNumCols());
    BOOST_CHECK_EQUAL(0, matrixA.GetNumRows());

    const SingleMatrix matrixA1(c_deviceIdZero);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA1.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(0, matrixA1.GetNumCols());
    BOOST_CHECK_EQUAL(0, matrixA1.GetNumRows());

    const SingleMatrix matrixA2(static_cast<size_t>(13), static_cast<size_t>(12), c_deviceIdZero);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA2.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(12, matrixA2.GetNumCols());
    BOOST_CHECK_EQUAL(13, matrixA2.GetNumRows());

    float arr[5 * 45];
    const SingleMatrix matrixA3(5, 45, arr, c_deviceIdZero, matrixFlagNormal);
    BOOST_CHECK(CurrentDataLocation::GPU == matrixA3.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(45, matrixA3.GetNumCols());
    BOOST_CHECK_EQUAL(5, matrixA3.GetNumRows());
}

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_AccessPatternAndTransferTest, RandomSeedFixture)
{
    float arr[5 * 45];
    const SingleMatrix matrixA(5, 45, arr, c_deviceIdZero, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(45, matrixA.GetNumCols());
    BOOST_CHECK_EQUAL(5, matrixA.GetNumRows());

    float x = matrixA(1, 1);
    BOOST_CHECK(matrixA.GetCurrentMatrixLocation() != CurrentDataLocation::GPU);
    foreach_coord (i, j, matrixA)
    {
        x = matrixA(i, j);
        BOOST_CHECK(matrixA.GetCurrentMatrixLocation() != CurrentDataLocation::GPU);
    }

    SingleMatrix matrixB(15, 15, arr, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(15, matrixB.GetNumCols());
    BOOST_CHECK_EQUAL(15, matrixB.GetNumRows());

    x = matrixB(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    matrixB(4, 2) = x;
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    foreach_coord (i, j, matrixB)
    {
        x = matrixB(i, j);
        matrixB(j, i) = x;
        BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());
    }

    matrixB.TransferFromDeviceToDevice(-1, 0, false);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(0, -1, false);
    BOOST_CHECK_EQUAL(CurrentDataLocation::BOTH, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(-1, 0, true);
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());
    matrixB.TransferFromDeviceToDevice(0, -1, true);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());

    const SingleMatrix matrixC(-1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixC.GetCurrentMatrixLocation());

    const SingleMatrix matrixD(15, 15, arr, CPUDEVICE, matrixFlagNormal);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixD.GetCurrentMatrixLocation());
}

// Requires GPU
BOOST_FIXTURE_TEST_CASE(MatrixDataSynchronization_GravitatingTowardsPreferredDevice, RandomSeedFixture)
{
    SingleMatrix matrixA = SingleMatrix::RandomGaussian(64, 23, c_deviceIdZero, 0, 2, IncrementCounter());
    SingleMatrix matrixB = SingleMatrix::Eye(23, c_deviceIdZero);

    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());

    // Set the current matrix location by reading a value of the matrix
    float x = matrixA(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixA.GetCurrentMatrixLocation());
    x = matrixB(1, 1);
    BOOST_CHECK_EQUAL(CurrentDataLocation::CPU, matrixB.GetCurrentMatrixLocation());

    const SingleMatrix matrixC = matrixA * matrixB;
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixA.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixB.GetCurrentMatrixLocation());
    BOOST_CHECK_EQUAL(CurrentDataLocation::GPU, matrixC.GetCurrentMatrixLocation());
}

#endif

BOOST_AUTO_TEST_SUITE_END()
}
} } }
