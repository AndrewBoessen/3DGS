// trainer.hpp

#pragma once

#include "dataloader/colmap.hpp"
#include "gsplat/gaussian.hpp"
#include "gsplat/optimize.hpp"
#include "gsplat/utils.hpp"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

class Trainer {
public:
  Trainer() = default;

  Trainer(ConfigParameters config, Gaussians gaussians, std::unordered_map<int, Image> images,
          std::unordered_map<int, Camera> cameras)
      : config(std::move(config)), gaussians(std::move(gaussians)), images(std::move(images)),
        cameras(std::move(cameras)), optimizer(config.base_lr) {}

  void test_train_split();

  void reset_opacity();

  void add_sh_band();

  void adaptive_density();

  void train();

private:
  ConfigParameters config;
  Gaussians gaussians;
  std::unordered_map<int, Image> images;
  std::unordered_map<int, Camera> cameras;

  std::vector<Image> test_images;
  std::vector<Image> train_images;

  AdamOptimizer optimizer;

  int iter = 0;

  void split_gaussians(const std::vector<bool> &split_mask);

  void clone_gaussians(const std::vector<bool> &clone_mask);
};
