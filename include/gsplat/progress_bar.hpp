#pragma once

#include <chrono>
#include <iostream>
#include <string>

class ProgressBar {
public:
  ProgressBar(int total_iterations, int bar_width = 50)
      : total_iters(total_iterations), width(bar_width), current_iter(0) {
    start_time = std::chrono::high_resolution_clock::now();
    display();
  }

  void update(int current_iteration, float loss, int num_gaussians) {
    current_iter = current_iteration;
    current_loss = loss;
    current_num_gaussians = num_gaussians;
    display();
  }

  void finish() {
    current_iter = total_iters; // Ensure it shows 100%
    display();
    std::cerr << std::endl; // New line after completion
  }

private:
  int total_iters;
  int width;
  int current_iter;
  float current_loss = 0.0f;
  int current_num_gaussians = 0;
  std::chrono::high_resolution_clock::time_point start_time;

  void display() {
    float progress = static_cast<float>(current_iter) / total_iters;
    int num_chars = static_cast<int>(progress * width);

    std::string bar = "[";
    for (int i = 0; i < width; ++i) {
      if (i < num_chars) {
        bar += "=";
      } else {
        bar += " ";
      }
    }
    bar += "]";

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;

    std::cerr << "\r" << bar << " " << static_cast<int>(progress * 100.0) << "% "
              << "Iter: " << current_iter << "/" << total_iters << " Loss: " << std::setprecision(4) << current_loss
              << " Gaussians: " << current_num_gaussians << " Elapsed: " << static_cast<int>(elapsed.count()) << "s"
              << std::flush;
  }
};
