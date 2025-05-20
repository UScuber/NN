from sklearn.datasets import load_iris, load_digits, load_wine

def save_dataset(filename, load_fn):
  dataset = load_fn()

  with open(filename, "w") as f:
    f.write("\n".join([" ".join([str(x) for x in row] + [str(ans)]) for row, ans in zip(dataset.data, dataset.target)]))
    f.write("\n")
  

save_dataset("iris.txt", load_iris)
save_dataset("digits.txt", load_digits)
save_dataset("wine.txt", load_wine)
