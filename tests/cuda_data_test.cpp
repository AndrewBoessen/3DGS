#include "gtest/gtest.h"
#include <numeric>
#include <thrust/host_vector.h>

template <int D, typename T>
thrust::host_vector<T> compact_masked_array(const thrust::host_vector<T> &source, const thrust::host_vector<bool> &mask,
                                            int num_to_keep) {
  thrust::host_vector<T> compacted(num_to_keep * D);
  int compacted_idx = 0;
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) {
      for (int j = 0; j < D; ++j) {
        compacted[compacted_idx++] = source[i * D + j];
      }
    }
  }
  return compacted;
}

template <int D, typename T>
void scatter_masked_array(const thrust::host_vector<T> &source, const thrust::host_vector<bool> &mask,
                          thrust::host_vector<T> &destination) {
  int source_idx = 0;
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) {
      for (int j = 0; j < D; ++j) {
        destination[i * D + j] = source[source_idx++];
      }
    }
  }
}

class CudaDataTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

TEST_F(CudaDataTest, CompactMaskedArrayStride1) {
  thrust::host_vector<float> h_source = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  thrust::host_vector<bool> h_mask = {true, false, true, false, true};
  int num_culled = 3;
  thrust::host_vector<float> compacted = compact_masked_array<1>(h_source, h_mask, num_culled);
  thrust::host_vector<float> h_expected = {1.0f, 3.0f, 5.0f};
  ASSERT_EQ(compacted, h_expected);
}

TEST_F(CudaDataTest, CompactMaskedArrayStride3) {
  thrust::host_vector<float> h_source = {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f, 3.0f, 3.1f, 3.2f, 4.0f, 4.1f, 4.2f};
  thrust::host_vector<bool> h_mask = {true, false, true, false};
  int num_culled = 2;
  thrust::host_vector<float> compacted = compact_masked_array<3>(h_source, h_mask, num_culled);
  thrust::host_vector<float> h_expected = {1.0f, 1.1f, 1.2f, 3.0f, 3.1f, 3.2f};
  ASSERT_EQ(compacted, h_expected);
}

TEST_F(CudaDataTest, CompactMaskedArrayEmpty) {
  thrust::host_vector<float> h_source;
  thrust::host_vector<bool> h_mask;
  int num_culled = 0;
  thrust::host_vector<float> compacted = compact_masked_array<3>(h_source, h_mask, num_culled);
  thrust::host_vector<float> h_expected;
  ASSERT_EQ(compacted, h_expected);
}

TEST_F(CudaDataTest, CompactMaskedArrayAllTrue) {
  thrust::host_vector<float> h_source = {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f};
  thrust::host_vector<bool> h_mask = {true, true};
  int num_culled = 2;
  thrust::host_vector<float> compacted = compact_masked_array<3>(h_source, h_mask, num_culled);
  ASSERT_EQ(compacted, h_source);
}

TEST_F(CudaDataTest, CompactMaskedArrayAllFalse) {
  thrust::host_vector<float> h_source = {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f};
  thrust::host_vector<bool> h_mask = {false, false};
  int num_culled = 0;
  thrust::host_vector<float> compacted = compact_masked_array<3>(h_source, h_mask, num_culled);
  thrust::host_vector<float> h_expected;
  ASSERT_EQ(compacted, h_expected);
}

TEST_F(CudaDataTest, ScatterMaskedArrayStride1) {
  thrust::host_vector<float> h_compacted = {1.0f, 3.0f, 5.0f};
  thrust::host_vector<bool> h_mask = {true, false, true, false, true};
  thrust::host_vector<float> destination(5);
  scatter_masked_array<1>(h_compacted, h_mask, destination);
  thrust::host_vector<float> h_expected = {1.0f, 0.0f, 3.0f, 0.0f, 5.0f};
  ASSERT_EQ(destination, h_expected);
}

TEST_F(CudaDataTest, ScatterMaskedArrayStride3) {
  thrust::host_vector<float> h_compacted = {1.0f, 1.1f, 1.2f, 3.0f, 3.1f, 3.2f};
  thrust::host_vector<bool> h_mask = {true, false, true, false};
  thrust::host_vector<float> destination(12);
  scatter_masked_array<3>(h_compacted, h_mask, destination);
  thrust::host_vector<float> h_expected = {1.0f, 1.1f, 1.2f, 0.0f, 0.0f, 0.0f, 3.0f, 3.1f, 3.2f, 0.0f, 0.0f, 0.0f};
  ASSERT_EQ(destination, h_expected);
}

TEST_F(CudaDataTest, ScatterMaskedArrayEmpty) {
  thrust::host_vector<float> h_compacted;
  thrust::host_vector<bool> h_mask;
  thrust::host_vector<float> destination(0);
  scatter_masked_array<3>(h_compacted, h_mask, destination);
  thrust::host_vector<float> h_expected;
  ASSERT_EQ(destination, h_expected);
}

TEST_F(CudaDataTest, ScatterMaskedArrayAllTrue) {
  thrust::host_vector<float> h_compacted = {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f};
  thrust::host_vector<bool> h_mask = {true, true};
  thrust::host_vector<float> destination(6);
  scatter_masked_array<3>(h_compacted, h_mask, destination);
  ASSERT_EQ(destination, h_compacted);
}

TEST_F(CudaDataTest, ScatterMaskedArrayAllFalse) {
  thrust::host_vector<float> h_compacted;
  thrust::host_vector<bool> h_mask = {false, false};
  thrust::host_vector<float> destination(6);
  thrust::fill(destination.begin(), destination.end(), 42.0f);
  scatter_masked_array<3>(h_compacted, h_mask, destination);
  thrust::host_vector<float> h_expected = {42.0f, 42.0f, 42.0f, 42.0f, 42.0f, 42.0f};
  ASSERT_EQ(destination, h_expected);
}
