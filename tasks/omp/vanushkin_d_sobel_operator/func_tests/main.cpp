// Copyright 2024 Vanushkin Dmitry
#include <gtest/gtest.h>

#include <vector>

#include "omp/vanushkin_d_sobel_operator/include/sobel_operator_omp.hpp"

TEST(Vanushkin_D_ParSobelOperator, ApplySobelTo1x2000ColorizedImage) {
  size_t height = 1;
  size_t width = 2000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  SobelOperatorSequential seqTask(taskDataSeq);

  ASSERT_FALSE(seqTask.validation());

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs_count.emplace_back(width);
  taskDataParallel->inputs_count.emplace_back(height);
  taskDataParallel->outputs_count.emplace_back(width - 2);
  taskDataParallel->outputs_count.emplace_back(height - 2);

  SobelOperatorParallelOmp parTask(taskDataParallel);

  ASSERT_FALSE(parTask.validation());
}

TEST(Vanushkin_D_ParSobelOperator, ApplySobelTo10000x0ColorizedImage) {
  size_t height = 10000;
  size_t width = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  SobelOperatorSequential seqTask(taskDataSeq);

  ASSERT_FALSE(seqTask.validation());

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs_count.emplace_back(width);
  taskDataParallel->inputs_count.emplace_back(height);
  taskDataParallel->outputs_count.emplace_back(width - 2);
  taskDataParallel->outputs_count.emplace_back(height - 2);

  SobelOperatorParallelOmp parTask(taskDataParallel);

  ASSERT_FALSE(parTask.validation());
}

TEST(Vanushkin_D_ParSobelOperator, ApplySobelTo3x3ColorizedImage) {
  size_t height = 3;
  size_t width = 3;

  std::vector<Color> input = generate_image(width, height);

  std::vector<Grayscale> seqOut((height - 2) * (width - 2), 0);
  std::vector<Grayscale> parallOut((height - 2) * (width - 2), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqOut.data()));
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  SobelOperatorSequential seqTask(taskDataSeq);

  ASSERT_TRUE(seqTask.validation());
  ASSERT_TRUE(seqTask.pre_processing());
  ASSERT_TRUE(seqTask.run());
  ASSERT_TRUE(seqTask.post_processing());

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataParallel->inputs_count.emplace_back(width);
  taskDataParallel->inputs_count.emplace_back(height);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallOut.data()));
  taskDataParallel->outputs_count.emplace_back(width - 2);
  taskDataParallel->outputs_count.emplace_back(height - 2);

  SobelOperatorParallelOmp parTask(taskDataParallel);

  ASSERT_TRUE(parTask.validation());
  ASSERT_TRUE(parTask.pre_processing());
  ASSERT_TRUE(parTask.run());
  ASSERT_TRUE(parTask.post_processing());

  for (size_t i = 0; i < seqOut.size(); ++i) {
    ASSERT_EQ(seqOut[i], parallOut[i]);
  }
}

TEST(Vanushkin_D_ParSobelOperator, ApplySobelTo4x7ColorizedImage) {
  size_t height = 7;
  size_t width = 4;

  std::vector<Color> input = generate_image(width, height);

  std::vector<Grayscale> seqOut((height - 2) * (width - 2), 0);
  std::vector<Grayscale> parallOut((height - 2) * (width - 2), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqOut.data()));
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  SobelOperatorSequential seqTask(taskDataSeq);

  ASSERT_TRUE(seqTask.validation());
  ASSERT_TRUE(seqTask.pre_processing());
  ASSERT_TRUE(seqTask.run());
  ASSERT_TRUE(seqTask.post_processing());

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataParallel->inputs_count.emplace_back(width);
  taskDataParallel->inputs_count.emplace_back(height);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallOut.data()));
  taskDataParallel->outputs_count.emplace_back(width - 2);
  taskDataParallel->outputs_count.emplace_back(height - 2);

  SobelOperatorParallelOmp parTask(taskDataParallel);

  ASSERT_TRUE(parTask.validation());
  ASSERT_TRUE(parTask.pre_processing());
  ASSERT_TRUE(parTask.run());
  ASSERT_TRUE(parTask.post_processing());

  for (size_t i = 0; i < seqOut.size(); ++i) {
    ASSERT_EQ(seqOut[i], parallOut[i]);
  }
}

TEST(Vanushkin_D_ParSobelOperator, ApplySobelTo100x250ColorizedImage) {
  size_t height = 100;
  size_t width = 250;

  std::vector<Color> input = generate_image(width, height);

  std::vector<Grayscale> seqOut((height - 2) * (width - 2), 0);
  std::vector<Grayscale> parallOut((height - 2) * (width - 2), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqOut.data()));
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  SobelOperatorSequential seqTask(taskDataSeq);

  ASSERT_TRUE(seqTask.validation());
  ASSERT_TRUE(seqTask.pre_processing());
  ASSERT_TRUE(seqTask.run());
  ASSERT_TRUE(seqTask.post_processing());

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataParallel->inputs_count.emplace_back(width);
  taskDataParallel->inputs_count.emplace_back(height);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallOut.data()));
  taskDataParallel->outputs_count.emplace_back(width - 2);
  taskDataParallel->outputs_count.emplace_back(height - 2);

  SobelOperatorParallelOmp parTask(taskDataParallel);

  ASSERT_TRUE(parTask.validation());
  ASSERT_TRUE(parTask.pre_processing());
  ASSERT_TRUE(parTask.run());
  ASSERT_TRUE(parTask.post_processing());

  for (size_t i = 0; i < seqOut.size(); ++i) {
    ASSERT_EQ(seqOut[i], parallOut[i]);
  }
}

TEST(Vanushkin_D_ParSobelOperator, ApplySobelTo500x500ColorizedImage) {
  size_t height = 500;
  size_t width = 500;

  std::vector<Color> input = generate_image(width, height);

  std::vector<Grayscale> seqOut((height - 2) * (width - 2), 0);
  std::vector<Grayscale> parallOut((height - 2) * (width - 2), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqOut.data()));
  taskDataSeq->outputs_count.emplace_back(width - 2);
  taskDataSeq->outputs_count.emplace_back(height - 2);

  SobelOperatorSequential seqTask(taskDataSeq);

  ASSERT_TRUE(seqTask.validation());
  ASSERT_TRUE(seqTask.pre_processing());
  ASSERT_TRUE(seqTask.run());
  ASSERT_TRUE(seqTask.post_processing());

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataParallel->inputs_count.emplace_back(width);
  taskDataParallel->inputs_count.emplace_back(height);
  taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallOut.data()));
  taskDataParallel->outputs_count.emplace_back(width - 2);
  taskDataParallel->outputs_count.emplace_back(height - 2);

  SobelOperatorParallelOmp parTask(taskDataParallel);

  ASSERT_TRUE(parTask.validation());
  ASSERT_TRUE(parTask.pre_processing());
  ASSERT_TRUE(parTask.run());
  ASSERT_TRUE(parTask.post_processing());

  for (size_t i = 0; i < seqOut.size(); ++i) {
    ASSERT_EQ(seqOut[i], parallOut[i]);
  }
}
