import csv

f_path = "CNN_outputs_no_epoch.txt"
f = open(f_path, "r")
lines = f.readlines()
f.close()

runs = []
tmp = []
for i in lines:
    if i == "\n":
        runs.append(tmp)
        tmp = []
    else:
        tmp.append(i.strip())

print(runs[0])

# HEADERS: Model type:3, Batch size:4, Epochs:5, Learning rate:6, Device:8, Workers:9, Train:-2, Validation:-1
headers = ["Model type", "Batch size", "Epochs", "Learning rate", "Device", "Workers", "Train accuracy", "Train F1 score", "Train time", "Test accuracy", "Test F1 score"]
model_type = runs[0][3].split("|")[2].strip()
batch_size = runs[0][4].split("|")[2].strip()
epochs = runs[0][5].split("|")[2].strip()
learning_rate = runs[0][6].split("|")[2].strip()
device = runs[0][8].split("|")[2].strip()
workers = runs[0][9].split("|")[2].strip()
train = runs[0][-2].split(" ")
train_acc = train[4][:-1]
train_f1 = train[8]
train_time = train[-1][:-1]
test = runs[0][-1].split(" ")
test_acc = test[5][:-1]
test_f1 = test[9]
with open("CNN_outputs_no_epoch.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for run in runs:
        if len(run) > 0:
            model_type = run[3].split("|")[2].strip()
            batch_size = run[4].split("|")[2].strip()
            epochs = run[5].split("|")[2].strip()
            learning_rate = run[6].split("|")[2].strip()
            device = run[8].split("|")[2].strip()
            workers = run[9].split("|")[2].strip()
            train = run[-2].split(" ")
            train_acc = train[4][:-1]
            train_f1 = train[8]
            train_time = train[-1][:-1]
            test = run[-1].split(" ")
            test_acc = test[5][:-1]
            test_f1 = test[9]
            writer.writerow([model_type, batch_size, epochs, learning_rate, device, workers, train_acc, train_f1, train_time, test_acc, test_f1])
