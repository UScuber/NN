#pragma once
#include <random>
#include "layer.hpp"

struct Model {
  Model(const double lr=0.05) : rho(lr){}

  ~Model(){
    for(auto &l : layers) delete l;
  }

  template<typename T, typename = std::enable_if<std::is_base_of_v<Layer, T>>>
  void add_layer(const int size){
    if((int)layers.size() >= 1){
      W.push_back(matrix<double>(layers.back()->size, size));
    }
    layers.push_back(new T(size, layers.empty() ? 0 : layers.back()->size));
  }

  // 重みをランダムにする
  void init_params(){
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> dist(0.0, 1.0);
    for(auto &w : W){
      for(int i = 0; i < w.height(); i++){
        for(int j = 0; j < w.width(); j++){
          w[i][j] = dist(engine);
        }
      }
    }
  }

  void train(const matrix<double> &X, const matrix<double> &y, const int epochs){
    assert(X.height() == y.height() && y.height() >= 1 && epochs >= 1);
    assert((int)layers.size() >= 2);
    assert(layers[0]->size == X.width());
    assert(y.width() == layers.back()->size);

    for(int step = 0; step < epochs; step++){
      // online
      double avg_err = 0;
      for(int p = 0; p < y.height(); p++){
        std::vector<double> gl = forward(X[p]);
        assert((int)gl.size() == layers.back()->size);

        std::vector<double> e(layers.back()->size);
        for(int i = 0; i < layers.back()->size; i++){
          e[i] = gl[i] - y[p][i];
        }

        const double err = dot(e, e) / 2.0;
        avg_err += err;

        backward(e);
      }
      avg_err /= y.height();
      // if(step % 100 == 0)
      std::cerr << "Epoch: " << step << ", Loss: " << avg_err << "\n";
    }
  }

  // 最大値のindexを返す
  std::vector<int> predict(const matrix<double> &X){
    std::vector<int> result(X.height());
    for(int i = 0; i < X.height(); i++){
      const auto gl = forward(X[i]);
      assert((int)gl.size() == layers.back()->size);
      result[i] = std::max_element(gl.begin(), gl.end()) - gl.begin();
    }
    return result;
  }

  void show_overview() const{
    std::cout << "Model Overview: \n";
    for(const auto &l : layers){
      std::cout << "- " << l->name() << ": " << l->size << "\n";
    }
  }

private:
  std::vector<double> forward(const std::vector<double> &x){
    std::vector<double> g = x;
    for(int i = 0; i < (int)layers.size()-1; i++){
      g = layers[i+1]->forward(g, W[i]);
    }
    return g;
  }

  void backward(const std::vector<double> &el){
    std::vector<double> e = el;

    for(int i = (int)layers.size()-1; i >= 1; i--){
      // std::cerr << i << " " << e.size() << " " << layers[i]->size << "   ";
      assert(layers[i]->size == (int)e.size());
      e = layers[i]->backward(e, W[i-1], rho);
    }
  }

  std::vector<Layer*> layers;
  std::vector<matrix<double>> W;
  double rho;
};