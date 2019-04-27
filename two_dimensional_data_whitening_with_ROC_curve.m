% aliaksei petsiuk 2019
% Matlab 2016

%% Data set No. 3

clear all; clc;

% load two dimensional arrays of random data 
% for hypotheses H0 and H1

load('data3.mat');

figure
scatter(X0(1,:),X0(2,:),'b');
hold on;grid on;
scatter(X1(1,:),X1(2,:),'r');
legend('H0','H1');
hold on;

% assume that the given data are gaussian
% with unknown parameters.
% Use fitdist to estimate mean and std. dev.

pd_H0x_norm = fitdist(X0(1,:)','normal'); 
pd_H0y_norm = fitdist(X0(2,:)','normal');

pd_H1x_norm = fitdist(X1(1,:)','normal'); 
pd_H1y_norm = fitdist(X1(2,:)','normal');

% create circles for better illustration
th = linspace(0,2*pi,200);

% sf0 - tri-sigma
sf0 = 3;

% plot 3-sigma circles
xunit = sf0*pd_H0x_norm.sigma * cos(th) + mean(X0(1,:));
yunit = sf0*pd_H0y_norm.sigma * sin(th) + mean(X0(2,:));
plot(xunit, yunit,'b','LineWidth',1);hold on;

sf1 = 3;
xunit1 = sf1*pd_H1x_norm.sigma * cos(th) + mean(X1(1,:));
yunit1 = sf1*pd_H1y_norm.sigma * sin(th) + mean(X1(2,:));
plot(xunit1, yunit1,'r','LineWidth',1);hold off;
set(gca,'linewidth',1,'fontsize', 10);
title('Original Data Set');

% initial amount of data samples
fprintf('Initial X0 size: %g\n', size(X0,2));
fprintf('Initial X1 size: %g\n', size(X1,2));



%% Whiten the data 

mean_X1 = mean(X1,2);
mean_X0 = mean(X0,2);

covariance_X1 = cov(X1');
covariance_X0 = cov(X0');

Augmented_array = [X0, X1];

[U_X0, Lambda_X0] = eig(covariance_X0);

Z = (Lambda_X0)^(-1/2)*U_X0'*(Augmented_array - mean_X0);
% joint covariance
C = (Lambda_X0)^(-1/2)*U_X0'*covariance_X1*U_X0*(Lambda_X0)^(-1/2);

[U_X1, Lambda_X1] = eig(C);
Y = U_X1'*Z;
C1 = U_X1'*C*U_X1;

white_mean_1 = U_X1'*(Lambda_X0)^(-1/2)*U_X0'*(mean_X1-mean_X0);

dimension_size = size(X0,2);

pd_wH0x_norm = fitdist(Y(1,1:dimension_size)','normal'); 
pd_wH0y_norm = fitdist(Y(2,1:dimension_size)','normal');

pd_wH1x_norm = fitdist(Y(1,dimension_size+1:end)','normal'); 
pd_wH1y_norm = fitdist(Y(2,dimension_size+1:end)','normal');

% create circles for better illustration
th = linspace(0,2*pi,200);

figure
wxunit = 3*pd_wH0x_norm.sigma * cos(th) + pd_wH0x_norm.mu;
wyunit = 3*pd_wH0y_norm.sigma * sin(th) + pd_wH0y_norm.mu;
plot(wxunit, wyunit,'g','LineWidth',1);hold on;


wxunit1 = 3*pd_wH1x_norm.sigma * cos(th) + pd_wH1x_norm.mu;
wyunit1 = 3*pd_wH1y_norm.sigma * sin(th) + pd_wH1y_norm.mu;
plot(wxunit1, wyunit1,'m','LineWidth',1);hold off;
hold on;

scatter(Y(1,1:dimension_size),Y(2,1:dimension_size),'g')
scatter(Y(1,dimension_size+1:end),Y(2,dimension_size+1:end),'m')
muX0b = mean(Y(:,1:dimension_size),2);
muX1b = mean(Y(:,dimension_size+1:end),2);

legend('Whitened H0','Whitened H1');
grid on;
set(gca,'linewidth',1,'fontsize', 10);
title('Whitened Data Set');


%% Find the Receiver Operational Characteristic

Pd  = [];
Pfa = [];

for i=1:length(Y)
    % estimation threshold for gaussian with equal mean
    zeta_mu(i)= 2*white_mean_1(1)/C1(1,1)*Y(1,i)+2*white_mean_1(2)/C1(2,2)*Y(2,i);
    
    % estimation threshold for gaussian with equal variance
    zeta_sigma(i)=(C1(1,1)-1)/C1(1,1)*Y(1,i)^2+(C1(2,2)-1)/C1(2,2)*Y(2,i)^2;
    
    % estimation threshold for general case
    zeta(i)=zeta_mu(i)+zeta_sigma(i);
end

Augmented_array = sortrows([zeta;zeros(1,dimension_size),ones(1,dimension_size)]');

for i = 1:length(Augmented_array)
    Pd(i) = sum(Augmented_array(i:end,2))/dimension_size;
    Pfa(i) = (length(Augmented_array)-i+1-sum(Augmented_array(i:end,2)))/dimension_size;
end
Pd = flip([Pd,0]);
Pfa = flip([Pfa,0]);

figure
plot(Pfa',Pd','LineWidth',2);
xlabel('Pfa');ylabel('Pd');
grid on;
set(gca,'linewidth',1,'fontsize', 10);
title('ROC for data3');

fprintf('Pfa size: %g\n', size(Pfa,2));
fprintf('Pd size: %g\n', size(Pd,2));


%% Data set No. 4
%% Load and plot data
clear all; clc;

% load two dimensional arrays of random data 
% for hypotheses H0 and H1

load('data4.mat');
% set displaying properties for future figures


figure
scatter(X0(1,:),X0(2,:),'b');
hold on;grid on;
scatter(X1(1,:),X1(2,:),'r');
legend('H0','H1');
hold on;

% assume that the given data are gaussian
% with unknown parameters.
% Use fitdist to estimate mean and std. dev.

pd_H0x_norm = fitdist(X0(1,:)','normal'); 
pd_H0y_norm = fitdist(X0(2,:)','normal');

pd_H1x_norm = fitdist(X1(1,:)','normal'); 
pd_H1y_norm = fitdist(X1(2,:)','normal');

% create circles for better illustration
th = linspace(0,2*pi,200);

% sf0 - radius of the circle to remove outliers for H0
sf0 = 2.28;

xunit = sf0*pd_H0x_norm.sigma * cos(th) + mean(X0(1,:));
yunit = sf0*pd_H0y_norm.sigma * sin(th) + mean(X0(2,:));
plot(xunit, yunit,'b','LineWidth',1);hold on;

% sf1 - radius of the circle to remove outliers for H1
sf1 = 2.1;
xunit1 = sf1*pd_H1x_norm.sigma * cos(th) + mean(X1(1,:));
yunit1 = sf1*pd_H1y_norm.sigma * sin(th) + mean(X1(2,:));
plot(xunit1, yunit1,'r','LineWidth',1);hold off;
set(gca,'linewidth',1,'fontsize', 10);
title('Original Data Set');

% initial amount of data samples
fprintf('Initial X0 size: %g\n', size(X0,2));
fprintf('Initial X1 size: %g\n', size(X1,2));

% assign NaN for outlying element
for i=1:size(X0,2)
    if sqrt(X0(1,i).^2+X0(2,i).^2) >= sqrt(xunit(i).^2+yunit(i).^2)
        X0(1,i) = NaN;
        X0(2,i) = NaN;
    end
end

for i=1:size(X1,2)
    if sqrt(X1(1,i).^2+X1(2,i).^2) >= sqrt(xunit1(i).^2+yunit1(i).^2)
        X1(1,i) = NaN;
        X1(2,i) = NaN;
    end
end

% remove all NaN cells
out = X0(all(~isnan(X0),2),:); % for nan - rows
out = X0(:,all(~isnan(X0),1));   % for nan - columns

X0 = out;

out = X1(all(~isnan(X1),2),:); % for nan - rows
out = X1(:,all(~isnan(X1),1));   % for nan - columns

X1 = out(:,1:170);

fprintf('Final X0 size: %g\n', size(X0,2));
fprintf('Final X1 size: %g\n', size(X1,2));

%% Whiten the data
mean_X1 = mean(X1,2);
mean_X0 = mean(X0,2);

covariance_X1 = cov(X1');
covariance_X0 = cov(X0');

Augmented_array = [X0, X1];

[U_X0, Lambda_X0] = eig(covariance_X0);

Z = (Lambda_X0)^(-1/2)*U_X0'*(Augmented_array - mean_X0);
% joint covariance
C = (Lambda_X0)^(-1/2)*U_X0'*covariance_X1*U_X0*(Lambda_X0)^(-1/2);

[U_X1, Lambda_X1] = eig(C);
Y = U_X1'*Z;
C1 = U_X1'*C*U_X1;

white_mean_1 = U_X1'*(Lambda_X0)^(-1/2)*U_X0'*(mean_X1-mean_X0);

dimension_size = size(X0,2);

pd_wH0x_norm = fitdist(Y(1,1:dimension_size)','normal'); 
pd_wH0y_norm = fitdist(Y(2,1:dimension_size)','normal');

pd_wH1x_norm = fitdist(Y(1,dimension_size+1:end)','normal'); 
pd_wH1y_norm = fitdist(Y(2,dimension_size+1:end)','normal');

% create circles for better illustration
th = linspace(0,2*pi,200);

figure
wxunit = 3*pd_wH0x_norm.sigma * cos(th) + pd_wH0x_norm.mu;
wyunit = 3*pd_wH0y_norm.sigma * sin(th) + pd_wH0y_norm.mu;
plot(wxunit, wyunit,'g','LineWidth',1);hold on;


wxunit1 = 3*pd_wH1x_norm.sigma * cos(th) + pd_wH1x_norm.mu;
wyunit1 = 3*pd_wH1y_norm.sigma * sin(th) + pd_wH1y_norm.mu;
plot(wxunit1, wyunit1,'m','LineWidth',1);hold off;
hold on;

scatter(Y(1,1:dimension_size),Y(2,1:dimension_size),'g')
scatter(Y(1,dimension_size+1:end),Y(2,dimension_size+1:end),'m')
muX0b = mean(Y(:,1:dimension_size),2);
muX1b = mean(Y(:,dimension_size+1:end),2);

legend('Whitened H0','Whitened H1');
grid on;
set(gca,'linewidth',1,'fontsize', 10);
title('Whitened Data Set');


%% Calculate the Receiver Operational Characteristic
Pd  = [];
Pfa = [];

for i=1:length(Y)
    % estimation threshold for gaussian with equal mean
    zeta_mu(i)= 2*white_mean_1(1)/C1(1,1)*Y(1,i)+2*white_mean_1(2)/C1(2,2)*Y(2,i);
    
    % estimation threshold for gaussian with equal variance
    zeta_sigma(i)=(C1(1,1)-1)/C1(1,1)*Y(1,i)^2+(C1(2,2)-1)/C1(2,2)*Y(2,i)^2;
    
    % estimation threshold for general case
    zeta(i)=zeta_mu(i)+zeta_sigma(i);
end

Augmented_array = sortrows([zeta;zeros(1,dimension_size),ones(1,dimension_size)]');

for i = 1:length(Augmented_array)
    Pd(i) = sum(Augmented_array(i:end,2))/dimension_size;
    Pfa(i) = (length(Augmented_array)-i+1-sum(Augmented_array(i:end,2)))/dimension_size;
end
Pd = flip([Pd,0]);
Pfa = flip([Pfa,0]);

figure
plot(Pfa',Pd','LineWidth',2);
xlabel('Pfa');ylabel('Pd');
grid on;
set(gca,'linewidth',1,'fontsize', 10);
title('ROC for data4');

fprintf('Pfa size: %g\n', size(Pfa,2));
fprintf('Pd size: %g\n', size(Pd,2));

% --- end ---