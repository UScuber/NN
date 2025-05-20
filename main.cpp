#include "model.hpp"
#include "loaddataset.hpp"
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

int main(int argc, char *argv[]){
  std::string data_filename = "iris";
  if(argc >= 2){
    data_filename = std::string(argv[1]);
  }
  data_filename += ".txt";

  const double lr = 0.01;

  matrix<double> X_data;
  std::vector<int> y_data;
  load_dataset(X_data, y_data, data_filename);

  matrix<double> X_train, y_train, X_test;
  std::vector<int> y_test;
  train_test_split(X_train, y_train, X_test, y_test, X_data, y_data, 0.8);

  MinMaxScaler ss;
  ss.fit(X_train);
  ss.transform(X_train);
  ss.transform(X_test);

  Model model(lr);
  model.add_layer<Input>(X_train.width());
  model.add_layer<Sigmoid>(128);
  model.add_layer<Output>(y_train.width());
  model.show_overview();

  model.init_params();

  model.train(X_train, y_train, 40);


  const auto pred = model.predict(X_test);

  int ac = 0;
  for(int i = 0; i < X_test.height(); i++){
    if(y_test[i] == pred[i]) ac++;
  }
  std::cout << ac << "/" << X_test.height() << " = " << (double)ac / X_test.height() << "\n";
}
