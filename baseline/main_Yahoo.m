writecell({"dataset", "model", "F1", "k", "u", "c", "b"},'baseline_Yahoo.csv','Delimiter',',','QuoteStrings',1,"WriteMode","append")


files = dir('..\data\Yahoo\**\*.csv');
max_k = 20;
max_c = 50;
max_b = 50;
for i = 1:length(files)
    filename = strcat(files(i).folder, '\', files(i).name);

    data = readmatrix(filename);
    ts = data(:, 2)';
    correct = data(:, 3)';
    [acc, param, model] = models_tuning(ts, correct, max_k, max_c, max_b);

    dataset_name = strsplit(filename, "\");
    dataset_name = strjoin(dataset_name(end-1:end), "\");
    writecell({dataset_name, model, acc, param(1), param(2), param(3), param(4)},'baseline_Yahoo.csv','Delimiter',',','QuoteStrings',1,"WriteMode","append")

end