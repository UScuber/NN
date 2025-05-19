#include "model.hpp"
#include "irisdataset.hpp"
#include "scaler.hpp"


void train_test_split(
  matrix<double> &X_train,
  matrix<double> &y_train,
  matrix<double> &X_test,
  std::vector<int> &y_test,
  const matrix<double> &X_data,
  const std::vector<int> &y_data,
  const double train_size,
  const bool shuffle=true,
  const unsigned int seed=0 // 0:random seed
){
  assert(0 < train_size && train_size < 1);
  assert(X_data.height() == (int)y_data.size());

  std::random_device seed_gen;
  std::mt19937 engine(seed ? seed : seed_gen());

  std::vector<int> indices(X_data.height());
  std::iota(indices.begin(), indices.end(), 0);
  if(shuffle) std::shuffle(indices.begin(), indices.end(), engine);

  const int train_datasize = X_data.height() * train_size;
  const int testdatasize = X_data.height() - train_datasize;
  const std::vector<int> train_indices(indices.begin(), indices.begin() + train_datasize);
  const std::vector<int> test_indices(indices.begin() + train_datasize, indices.end());

  const int y_max = *std::max_element(y_data.begin(), y_data.end());

  X_train = matrix<double>(train_datasize, X_data.width());
  y_train = matrix<double>(train_datasize, y_max + 1);
  for(int i = 0; i < train_datasize; i++){
    X_train[i] = X_data[train_indices[i]];
    y_train[i][y_data[train_indices[i]]] = 1;
  }

  X_test = matrix<double>(testdatasize, X_data.width());
  y_test = std::vector<int>(testdatasize);
  for(int i = 0; i < testdatasize; i++){
    X_test[i] = X_data[test_indices[i]];
    y_test[i] = y_data[test_indices[i]];
  }
}

int main(){
  const double lr = 0.02;

  const int input_size = iris_dataset[0].data.size();

  Model model(lr);
  model.add_layer<Input>(input_size);
  model.add_layer<Sigmoid>(8);
  model.add_layer<Output>(3);

  model.init_params();
  // std::random_device seed_gen;
  // std::mt19937 engine(seed_gen());
  // std::shuffle(iris_dataset.begin(), iris_dataset.end(), engine);

  // const int datasize = (int)iris_dataset.size() * 0.8;
  // const int testdatasize = (int)iris_dataset.size() - datasize;
  // const std::vector<Iris> train_data(iris_dataset.begin(), iris_dataset.begin() + datasize);
  // const std::vector<Iris> test_data(iris_dataset.begin() + datasize, iris_dataset.end());

  // matrix<double> X_train(datasize, input_size);
  // matrix<double> y_train(datasize, 3);
  // for(int i = 0; i < datasize; i++){
  //   X_train[i] = iris_dataset[i].data;
  //   assert(iris_dataset[i].data.size() == 4);
  //   y_train[i][iris_dataset[i].kind] = 1;
  // }
  matrix<double> X_data;
  std::vector<int> y_data;
  load_iris(X_data, y_data);

  matrix<double> X_train, y_train, X_test;
  std::vector<int> y_test;
  train_test_split(X_train, y_train, X_test, y_test, X_data, y_data, 0.8);

  MinMaxScaler ss;
  ss.fit(X_train);
  ss.transform(X_train);
  ss.transform(X_test);

  model.train(X_train, y_train, 1000);


  const auto pred = model.predict(X_test);

  int ac = 0;
  for(int i = 0; i < X_test.height(); i++){
    if(y_test[i] == pred[i]) ac++;
  }
  std::cout << ac << "/" << X_test.height() << " = " << (double)ac / X_test.height() << "\n";
}
