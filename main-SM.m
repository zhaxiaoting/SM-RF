clc;
clear;
close all;
allData=xlsread('Required data.xlsx','Sheet1','A1:I142');
sortedIDX=xlsread('Required data.xlsx','Sheet1','I1:I142');
k=5;
[X_smote]= mySMOTE(allData,k,sortedIDX);