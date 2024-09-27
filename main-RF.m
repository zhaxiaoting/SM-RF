%%  Clear environment variables
warning off             % Turn off alarm information
close all               % Close the open image window
clear                   % Clear variables
clc                     % Clear Command Line


%% Data preprocessing
%load matlab777.mat;%Use this instruction for raw data
load matlab999.mat;%Use this command for oversampled data
%%  Divide the training set and testing set
%train_wine = [wine(1:142,:)];%Use this instruction for raw data
train_wine = [wine(1:356,:)];%Use this command for oversampled data

%train_wine_labels = [wine_labels(1:142);];%Use this instruction for raw data
train_wine_labels = [wine_labels(1:356);];%Use this command for oversampled data
%test_wine = [wine(143:203,:)];%Use this instruction for raw data
test_wine = [wine(357:508,:)];%Use this command for oversampled data

%test_wine_labels = [wine_labels(143:203,:)]; %Use this instruction for raw data
test_wine_labels = [wine_labels(357:508,:)];%Use this command for oversampled data
P_train = train_wine';
T_train = train_wine_labels';
M = size(P_train, 2);                  %size Return matrix size
P_test = test_wine';
T_test = test_wine_labels';
N = size(P_test, 2);

%%  data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  Transposition to adapt the model
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  training model
trees = 50;                                       % Number of decision trees
leaf  = 1;                                        % Minimum number of leaves
OOBPrediction = 'on';                             % Open the error chart
OOBPredictorImportance = 'on';                    % Calculate the importance of features
Method = 'classification';                        % Classification or Regression
net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance, ...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  % importance

%%  Simulation test
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  format conversion
T_sim1 = str2num(cell2mat(t_sim1));
T_sim2 = str2num(cell2mat(t_sim2));

%%  performance evaluation
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  Draw error curve
figure
plot(1 : trees, oobError(net), 'b-', 'LineWidth', 1)
legend('error curve')
xlabel('Number of decision trees')
ylabel('error')
xlim([1, trees])
grid

%%  Importance of drawing features
figure
bar(importance)
legend('Importance','fontsize',10)
xlabel('Characteristic','fontsize',10)
ylabel('Importance','fontsize',10)

%%  data sorting
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  draw
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('Actual value', 'Predictive value','fontsize',10)
xlabel('Training samples','fontsize',10)
ylabel('Predict category','fontsize',10)
string = {'Comparison of Training Set Prediction Results'; ['Accuracy Rate=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('Actual value', 'Predictive value','fontsize',10)
xlabel('Test samples','fontsize',10)
ylabel('Predict category','fontsize',10)
string = {'Comparison of Test Set Prediction Results'; ['Accuracy Rate=' num2str(error2) '%']};
title(string)
grid

%%  confusion matrix
%figure
%cm = confusionchart(T_train, T_sim1);
%cm.Title = 'Confusion Matrix for Train Data';
%cm.ColumnSummary = 'column-normalized';
%cm.RowSummary = 'row-normalized';
    
%figure
%cm = confusionchart(T_test, T_sim2);
%cm.Title = 'Confusion Matrix for Test Data';
%cm.ColumnSummary = 'column-normalized';
%cm.RowSummary = 'row-normalized';
