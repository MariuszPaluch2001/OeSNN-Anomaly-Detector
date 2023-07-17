function [best_acc, best_hyper, best_model] = models_tuning(TS, correct, k_max, c_max, b_max)
    %hyper parameters returned by tuning: k, u, c, b
    %if one of the parameters is unused, insert -1
    tunings = {};
    tunings{1} = @() outliner1_tuning(TS, correct, k_max, c_max, b_max);
    tunings{2} = @() outliner2_tuning(TS, correct, k_max, c_max, b_max);
    tunings{3} = @() outliner3_tuning(TS, correct, k_max, c_max, b_max);
    tunings{4} = @() outliner4_tuning(TS, correct, k_max, c_max, b_max);
    tunings{5} = @() outliner5_tuning(TS, correct, k_max, c_max, b_max);
    tunings{6} = @() outliner6_tuning(TS, correct, k_max, c_max, b_max);
    best_acc = 0;
    best_hyper = [0,0,0,0];
    best_model = 0;
    for i = 1:6
        [acc, hyper] = tunings{i}();
        if acc > best_acc
            best_acc = acc;
            best_hyper = hyper;
            best_model = i;
        end
    end
end

function [result] = calc_accuracy(preds, correct)
    correct = correct(1:end-1);
    tp = sum((preds == 1) & (correct == 1));
    fp = sum((preds == 1) & (correct == 0));
    fn = sum((preds == 0) & (correct == 1));
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = (2 * precision * recall) / (precision + recall);
    result = F1;
    %tp = length(preds(correct == 1));
    %fn = length(correct(preds == 0));
    %recall = sum(preds(correct == 1)) / sum(correct);
    %precission = sum(preds(preds == 1)) / sum(preds);
    %result = (2* recall * precission) / (recall + precission);
end

function [output] = oneliner1(TS, k, u, c, b)
    output = abs(diff(TS)) > u * movmean(abs(diff(TS)), k) + c * movstd(abs(diff(TS)), k) + b;
end

function [best_acc, hyper] = outliner1_tuning(TS, correct, k_max, c_max, b_max)
    best_acc = 0;
    best_k = 1;
    best_u = 0;
    best_c = 0;
    best_b = 0;
    for u = 0:1
        for k = 1:k_max
            for c = 0:c_max
                for b = 0:b_max
                    result = oneliner1(TS, k, u, c, b);
                    acc = calc_accuracy(result, correct);
                    if acc > best_acc
                        best_acc = acc;
                        best_k = k;
                        best_u = u;
                        best_c = c;
                        best_b = b;
                    end
                end
            end
        end
    end
    hyper = [best_k, best_u, best_c, best_b];
end

function [output] = oneliner2(TS, k, u, c, b)
    output = diff(TS) > u * movmean(diff(TS), k) + c * movstd(diff(TS), k) + b;
end

function [best_acc, hyper] = outliner2_tuning(TS, correct, k_max, c_max, b_max)
    best_acc = 0;
    best_k = 1;
    best_u = 0;
    best_c = 0;
    best_b = 0;
    for u = 0:1
        for k = 1:k_max
            for c = 0:c_max
                for b = 0:b_max
                    result = oneliner2(TS, k, u, c, b);
                    acc = calc_accuracy(result, correct);
                    if acc > best_acc
                        best_acc = acc;
                        best_k = k;
                        best_u = u;
                        best_c = c;
                        best_b = b;
                    end
                end
            end
        end
    end
    hyper = [best_k, best_u, best_c, best_b];
end

function [output] = oneliner3(TS, b)
    output = abs(diff(TS)) > b;
end


function [best_acc, hyper] = outliner3_tuning(TS, correct, ~, ~, b_max)
    best_acc = 0;
    best_b = 0;
    for b = 0:b_max
        result = oneliner3(TS, b);
        acc = calc_accuracy(result, correct);
        if acc > best_acc
            best_acc = acc;
            best_b = b;
        end
    end
    hyper = [-1, -1, -1, best_b];
end

function [output] = oneliner4(TS, k, c, b)
    output = abs(diff(TS)) > movmean(abs(diff(TS)), k) + c * movstd(abs(diff(TS)), k) + b;
end

function [best_acc, hyper] = outliner4_tuning(TS, correct, k_max, c_max, b_max)
    best_acc = 0;
    best_k = 1;
    best_c = 0;
    best_b = 0;
    for k = 1:k_max
        for c = 0:c_max
            for b = 0:b_max
                result = oneliner4(TS, k, c, b);
                acc = calc_accuracy(result, correct);
                if acc > best_acc
                    best_acc = acc;
                    best_k = k;
                    best_c = c;
                    best_b = b;
                end
            end
        end
    end
    hyper = [best_k, -1, best_c, best_b];
end

function [output] = oneliner5(TS, b)
    output = diff(TS) > b;
end

function [best_acc, hyper] = outliner5_tuning(TS, correct, ~, ~, b_max)
    best_acc = 0;
    best_b = 0;
    for b = 0:b_max
        result = oneliner5(TS, b);
        acc = calc_accuracy(result, correct);
        if acc > best_acc
            best_acc = acc;
            best_b = b;
        end
    end
    hyper = [-1, -1, -1, best_b];
end

function [output] = oneliner6(TS, k, c, b)
    output = diff(TS) > movmean(diff(TS), k) + c * movstd(diff(TS), k) + b;
end

function [best_acc, hyper] = outliner6_tuning(TS, correct, k_max, c_max, b_max)
    best_acc = 0;
    best_k = 1;
    best_c = 0;
    best_b = 0;
    for k = 1:k_max
        for c = 0:c_max
            for b = 0:b_max
                result = oneliner6(TS, k, c, b);
                acc = calc_accuracy(result, correct);
                if acc > best_acc
                    best_acc = acc;
                    best_k = k;
                    best_c = c;
                    best_b = b;
                end
            end
        end
    end
    hyper = [best_k, -1, best_c, best_b];
end