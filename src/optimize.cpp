// optimize.cpp

#include "gsplat/optimize.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

AdamOptimizer::AdamOptimizer(float lr, float beta1, float beta2, float epsilon)
    : lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

void AdamOptimizer::initialize_states_if_needed(const Gaussians &params) {
  if (t_ > 0 || params.size() == 0) {
    // Already initialized or nothing to initialize.
    return;
  }

  const size_t n = params.size();

  // Initialize first and second moment vectors with zeros.
  state_.m_xyz.assign(n, Eigen::Vector3f::Zero());
  state_.v_xyz.assign(n, Eigen::Vector3f::Zero());
  state_.m_rgb.assign(n, Eigen::Vector3f::Zero());
  state_.v_rgb.assign(n, Eigen::Vector3f::Zero());
  state_.m_opacity.assign(n, 0.0f);
  state_.v_opacity.assign(n, 0.0f);
  state_.m_scale.assign(n, Eigen::Vector3f::Zero());
  state_.v_scale.assign(n, Eigen::Vector3f::Zero());
  state_.m_quaternion.assign(n, Eigen::Vector4f::Zero());
  state_.v_quaternion.assign(n, Eigen::Vector4f::Zero());

  if (params.sh.has_value()) {
    const auto &sh_params = *params.sh;
    state_.m_sh.emplace();
    state_.v_sh.emplace();
    state_.m_sh->reserve(n);
    state_.v_sh->reserve(n);
    for (const auto &sh_p : sh_params) {
      state_.m_sh->push_back(Eigen::VectorXf::Zero(sh_p.size()));
      state_.v_sh->push_back(Eigen::VectorXf::Zero(sh_p.size()));
    }
  }
}

void AdamOptimizer::step(Gaussians &params, const Gradients &grads, const std::vector<char> &mask) {
  // Lazily initialize optimizer state on the first call to step().
  initialize_states_if_needed(params);

  const size_t n = params.size();
  if (n == 0)
    return;

  assert(params.size() == mask.size() && "Mask and parameter sizes must match.");

  t_++; // Increment timestep

  int i = 0;

  // Compute bias correction factors
  const float m_hat_corr = 1.0f / (1.0f - std::pow(beta1_, t_));
  const float v_hat_corr = 1.0f / (1.0f - std::pow(beta2_, t_));

  for (size_t j = 0; j < n; ++j) {
    if ((bool)mask[j] == false)
      continue;
    // --- Update XYZ ---
    state_.m_xyz[i] = beta1_ * state_.m_xyz[i] + (1 - beta1_) * grads.xyz[i];
    state_.v_xyz[i] = beta2_ * state_.v_xyz[i] + (1 - beta2_) * grads.xyz[i].cwiseProduct(grads.xyz[i]);
    Eigen::Vector3f m_hat_xyz = state_.m_xyz[i] * m_hat_corr;
    Eigen::Vector3f v_hat_xyz = state_.v_xyz[i] * v_hat_corr;
    params.xyz[i] -= lr_ * m_hat_xyz.cwiseQuotient((v_hat_xyz.cwiseSqrt().array() + epsilon_).matrix());

    // --- Update RGB ---
    state_.m_rgb[i] = beta1_ * state_.m_rgb[i] + (1 - beta1_) * grads.rgb[i];
    state_.v_rgb[i] = beta2_ * state_.v_rgb[i] + (1 - beta2_) * grads.rgb[i].cwiseProduct(grads.rgb[i]);
    Eigen::Vector3f m_hat_rgb = state_.m_rgb[i] * m_hat_corr;
    Eigen::Vector3f v_hat_rgb = state_.v_rgb[i] * v_hat_corr;
    params.rgb[i] -= lr_ * m_hat_rgb.cwiseQuotient((v_hat_rgb.cwiseSqrt().array() + epsilon_).matrix());

    // --- Update Opacity (scalar) ---
    state_.m_opacity[i] = beta1_ * state_.m_opacity[i] + (1 - beta1_) * grads.opacity[i];
    state_.v_opacity[i] = beta2_ * state_.v_opacity[i] + (1 - beta2_) * (grads.opacity[i] * grads.opacity[i]);
    float m_hat_opacity = state_.m_opacity[i] * m_hat_corr;
    float v_hat_opacity = state_.v_opacity[i] * v_hat_corr;
    params.opacity[i] -= lr_ * m_hat_opacity / (std::sqrt(v_hat_opacity) + epsilon_);

    // --- Update Scale ---
    state_.m_scale[i] = beta1_ * state_.m_scale[i] + (1 - beta1_) * grads.scale[i];
    state_.v_scale[i] = beta2_ * state_.v_scale[i] + (1 - beta2_) * grads.scale[i].cwiseProduct(grads.scale[i]);
    Eigen::Vector3f m_hat_scale = state_.m_scale[i] * m_hat_corr;
    Eigen::Vector3f v_hat_scale = state_.v_scale[i] * v_hat_corr;
    params.scale[i] -= lr_ * m_hat_scale.cwiseQuotient((v_hat_scale.cwiseSqrt().array() + epsilon_).matrix());

    // --- Update Quaternion ---
    state_.m_quaternion[i] = beta1_ * state_.m_quaternion[i] + (1 - beta1_) * grads.quaternion[i];
    state_.v_quaternion[i] =
        beta2_ * state_.v_quaternion[i] + (1 - beta2_) * grads.quaternion[i].cwiseProduct(grads.quaternion[i]);
    Eigen::Vector4f m_hat_quat = state_.m_quaternion[i] * m_hat_corr;
    Eigen::Vector4f v_hat_quat = state_.v_quaternion[i] * v_hat_corr;
    params.quaternion[i].coeffs() -=
        lr_ * m_hat_quat.cwiseQuotient((v_hat_quat.cwiseSqrt().array() + epsilon_).matrix());
    // Eigen::Vector3f update_vec = m_hat_quat.cwiseQuotient((v_hat_quat.cwiseSqrt().array() + epsilon_).matrix());

    // float angle = lr_ * update_vec.norm();
    // if (angle > 0.0f) {
    //   Eigen::Vector3f axis = update_vec.normalized();
    //   Eigen::Quaternionf dq(Eigen::AngleAxisf(angle, axis));
    //   params.quaternion[i] = (dq * params.quaternion[i]).normalized();
    // }

    // --- Update SH (if they exist) ---
    if (params.sh.has_value() && grads.sh.has_value() && state_.m_sh.has_value()) {
      auto &m_sh_vec = *state_.m_sh;
      auto &v_sh_vec = *state_.v_sh;
      auto &param_sh_vec = *params.sh;
      const auto &grad_sh_vec = *grads.sh;

      m_sh_vec[i] = beta1_ * m_sh_vec[i] + (1 - beta1_) * grad_sh_vec[i];
      v_sh_vec[i] = beta2_ * v_sh_vec[i] + (1 - beta2_) * grad_sh_vec[i].cwiseProduct(grad_sh_vec[i]);
      Eigen::VectorXf m_hat_sh = m_sh_vec[i] * m_hat_corr;
      Eigen::VectorXf v_hat_sh = v_sh_vec[i] * v_hat_corr;
      param_sh_vec[i] -= lr_ * m_hat_sh.cwiseQuotient((v_hat_sh.cwiseSqrt().array() + epsilon_).matrix());
    }
    i++;
  }
}

void AdamOptimizer::filter_states(const std::vector<bool> &mask) {
  const size_t original_size = state_.m_xyz.size();
  if (original_size == 0)
    return;
  assert(mask.size() == original_size && "Mask size must match optimizer state size.");

  auto filter = [&](auto &vec) {
    if (vec.empty())
      return;
    size_t write_idx = 0;
    for (size_t read_idx = 0; read_idx < original_size; ++read_idx) {
      if (mask[read_idx]) {
        if (write_idx != read_idx) {
          vec[write_idx] = std::move(vec[read_idx]);
        }
        write_idx++;
      }
    }
    vec.resize(write_idx);
  };

  filter(state_.m_xyz);
  filter(state_.v_xyz);
  filter(state_.m_rgb);
  filter(state_.v_rgb);
  filter(state_.m_opacity);
  filter(state_.v_opacity);
  filter(state_.m_scale);
  filter(state_.v_scale);
  filter(state_.m_quaternion);
  filter(state_.v_quaternion);

  if (state_.m_sh.has_value()) {
    filter(*state_.m_sh);
    filter(*state_.v_sh);
  }
}

void AdamOptimizer::append_states(size_t n) {
  if (n == 0)
    return;

  // Determine the dimension of SH coefficients from existing states.
  long sh_dim = 0;
  if (state_.m_sh.has_value() && !state_.m_sh->empty()) {
    sh_dim = (*state_.m_sh)[0].size();
  }

  state_.m_xyz.insert(state_.m_xyz.end(), n, Eigen::Vector3f::Zero());
  state_.v_xyz.insert(state_.v_xyz.end(), n, Eigen::Vector3f::Zero());
  state_.m_rgb.insert(state_.m_rgb.end(), n, Eigen::Vector3f::Zero());
  state_.v_rgb.insert(state_.v_rgb.end(), n, Eigen::Vector3f::Zero());
  state_.m_opacity.insert(state_.m_opacity.end(), n, 0.0f);
  state_.v_opacity.insert(state_.v_opacity.end(), n, 0.0f);
  state_.m_scale.insert(state_.m_scale.end(), n, Eigen::Vector3f::Zero());
  state_.v_scale.insert(state_.v_scale.end(), n, Eigen::Vector3f::Zero());
  state_.m_quaternion.insert(state_.m_quaternion.end(), n, Eigen::Vector4f::Zero());
  state_.v_quaternion.insert(state_.v_quaternion.end(), n, Eigen::Vector4f::Zero());

  if (state_.m_sh.has_value()) {
    assert(sh_dim > 0 && "Cannot append SH states if initial SH dimension is unknown.");
    state_.m_sh->insert(state_.m_sh->end(), n, Eigen::VectorXf::Zero(sh_dim));
    state_.v_sh->insert(state_.v_sh->end(), n, Eigen::VectorXf::Zero(sh_dim));
  }
}

void AdamOptimizer::upgrade_sh_states(size_t num_gaussians, size_t new_sh_coeffs_count) {
  // Helper lambda to resize and update a single state vector
  auto upgrade_vector = [&](std::optional<std::vector<Eigen::VectorXf>> &vec) {
    if (!vec.has_value()) {
      // Initialize the state vector for the first time
      vec.emplace(num_gaussians, Eigen::VectorXf::Zero(new_sh_coeffs_count));
    } else {
      // Upgrade existing state vector
      auto &state_vec = vec.value();
      if (state_vec.empty() || state_vec[0].size() >= new_sh_coeffs_count) {
        return; // No upgrade needed or invalid state
      }
      size_t old_sh_coeffs_count = state_vec[0].size();
      for (auto &v : state_vec) {
        v.conservativeResize(new_sh_coeffs_count);
        v.tail(new_sh_coeffs_count - old_sh_coeffs_count).setZero();
      }
    }
  };

  // Upgrade both the first (m) and second (v) moment vectors for SH
  upgrade_vector(state_.m_sh);
  upgrade_vector(state_.v_sh);
}
