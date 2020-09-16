clc;
clear;
close all
datas=xlsread('dataxiu.xlsx');
labels=xlsread('label_ti.xlsx');
datas_labels = labels;

%% ���� 19��SNP������䡢����������
datas_b=datas(1:141,[1:3,5:20,22]);
label_b=datas_labels(1:141,:);
datas_h=datas(142:end,[1:3,5:20,22]);
label_h=datas_labels(142:end,:);

%% KS��������
m=size(datas_b,1);
n=size(datas_h,1)
[Train,dminmax] = KS(datas_b,ceil(m*0.7));
Preind = setdiff(1:size(datas_b,1),Train);
t_train=label_b(Train,:); t_test = label_b(Preind,:); %%��ǩ����
p_train = datas_b(Train,:);  p_test = datas_b(Preind,:); %%���ݻ���
[Test,dminmax_t] = KS(datas_h,ceil(n*0.7));
Preind_t = setdiff(1:size(datas_h,1),Test);
tt_train=label_h(Test,:); tt_test = label_h(Preind_t,:); %%��ǩ����
pp_train = datas_h(Test,:);  pp_test = datas_h(Preind_t,:); %%���ݻ���
train_wine =[ p_train;pp_train];
train_wine_labels =[t_train;tt_train];
test_wine = [p_test;pp_test];
test_wine_labels =[t_test;tt_test];
tra=[Train';Test'];
tes=[Preind';Preind_t'];


 %% ��һ�š�SVM + Ѱ��*4
[bestacc,bestc,bestg] = SVMcgForClass(train_wine_labels,train_wine,-10,10,-10,10,5,0.5,0.5,0.01);
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 0 -t 2 -p 0.1'];
model = svmtrain(train_wine_labels,train_wine,cmd);
[predict_label_train, accuracy_train,decision_values_train] = svmpredict(train_wine_labels, train_wine, model,'-b 0');
[predict_label, accuracy,decision_values] = svmpredict(test_wine_labels, test_wine, model,'-b 0');

number_a_sim = length(find(predict_label_train == 1 & train_wine_labels == 1));%%TP
number_aa_sim = length(find(predict_label_train == 0 &train_wine_labels == 0));%%TN
number_a = 110-number_aa_sim;%FP 
number_aa= 99-number_a_sim;%%FN

number_b_sim = length(find(predict_label == 1 & test_wine_labels == 1));%TP
number_bb_sim = length(find(predict_label == 0 &test_wine_labels == 0));%TN
number_b = 46-number_bb_sim;%FP
number_bb= 42-number_b_sim;%%FN

P_train=number_a_sim/(number_a_sim+number_a );
sen_train=number_a_sim/(number_a_sim+number_aa);
spe_train=number_aa_sim/(number_aa_sim+number_a);

train_result=[P_train,sen_train,spe_train]

P_test=number_b_sim/(number_b_sim+number_b);
sen_test=number_b_sim/(number_b_sim+number_bb);
spe_test=number_bb_sim/(number_bb_sim+number_b);

test_result=[P_test,sen_test,spe_test]

figure;
hold on;
plot(test_wine_labels,'o');
plot(predict_label,'r*');
set(gca,'yticklabel',{' ','1',' ','2 ',' '});
xlabel('����','FontSize',12);
ylabel('���','FontSize',12);
legend('ʵ�ʷ���','Ԥ�����');

snapnow;

