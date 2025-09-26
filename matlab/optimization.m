%% MARKOWITZ PORTFOLIO OPTIMIZATION
%
% MATLAB: R2024b
%
%
% INPUTS:
% - prices.csv
% - ff3.csv
%
% OUTPUTS:
% - outputs/figures/*.png (frontier, cml, weights)
% - outputs/tables/*.csv (regression table, mu, weights, datapoints)
%


%% FLAGS AND VARIABLES


prices_file = 'prices.csv';
ff3_file = 'ff3.csv';

frequency = 252;          % frequency of observations
include_alpha = false;    % include alpha in ff3 return computation

export_flag = true;       % save figures and tables
cutoff = 20;              % cutoff for portfolio composition graphs


%% DATA IMPORT


warning('off','MATLAB:table:ModifiedAndSavedVarnames')

stocks_data = readtable(prices_file, 'PreserveVariableNames', true);
ff = readtable(ff3_file);

if ismember('Mkt-RF', ff.Properties.VariableNames)
    ff.Properties.VariableNames{'Mkt-RF'} = 'Mkt_RF';
end


dates = stocks_data{:,1};
tickers = stocks_data.Properties.VariableNames(2:end);
prices = stocks_data{:,2:end};


%% DATA CLEANING


if ~isdatetime(dates)
    dates = datetime(dates);
end

if ~isdatetime(ff.Date)
    ff.Date = datetime(ff.Date);
end


S = array2table(prices, 'VariableNames', tickers);
S = addvars(S, dates, 'Before', 1, 'NewVariableNames', 'Date');

S = sortrows(S, 'Date');
ff = sortrows(ff,'Date');

[~, iu] = unique(S.Date, 'stable'); S = S(iu,:);
[~, iu] = unique(ff.Date, 'stable'); ff = ff(iu,:);


%% RETURN COMPUTATION


clean_prices = S{:,2:end};

R = clean_prices(2:end,:) ./ clean_prices(1:end-1,:) - 1;
dates = S.Date(2:end);

S_returns = array2table(R, 'VariableNames', tickers);
S_returns = addvars(S_returns, dates, 'Before', 1, 'NewVariableNames', 'Date');


%% DATE MATCHING THROUGH INNERJOIN


J = innerjoin(S_returns, ff, 'Keys', 'Date');
J = rmmissing(J);

Mkt_RF = J.Mkt_RF;
SMB = J.SMB;
HML = J.HML;
RF = J.RF;
dates  = J.Date;


R = J{:, tickers};
[T, N] = size(R);


%% EXPECTED RETURN WITH FAMA&FRENCH


Y = R - RF;
X = [ones(height(J),1), Mkt_RF, SMB, HML];


alpha = zeros(N,1);
betas = zeros(N,3);
R2 = zeros(N,1);


for i=1:N
    [b,~,~,~,stats] = regress(Y(:,i), X);
    alpha(i) = b(1);
    betas(i,:) = b(2:4)';
    R2(i) = stats(1);
end


lambda  = mean(X(:,2:4),1)';
Rf_mean = mean(J.RF);


mu_daily = Rf_mean + betas * lambda + (include_alpha * alpha);
mu = (1 + mu_daily).^frequency - 1;


%% EFFICIENT FRONTIER AND GMV PORTFOLIO


V_daily = cov(R);
V = frequency*V_daily;

e = ones(N,1);
A = [mu e];
B = A' * (V \ A);
a = B(1,1); b = B(1,2); c = B(2,2);

m = linspace(min(mu)-0.04, 0.5, 300);
sigma2 = (c*m.^2 - 2*b*m + a)/(a*c - b^2);
sigma2 = max(sigma2, 0);  % clamp
sigma = sqrt(sigma2);

x_GMVP = (V \ e)/c;
m_GMVP = b/c;
sigma2_GMVP = 1/c;
sigma_GMVP = sqrt(sigma2_GMVP);


%% GMV PORTFOLIO COMPOSITION



long_GMV = x_GMVP > 0;
wL_GMV = x_GMVP(long_GMV);
nL_GMV = tickers(long_GMV);

[wL_GMV, indexL_GMV] = sort(wL_GMV, 'descend');
nL_GMV = nL_GMV(indexL_GMV);

kL_GMV = min(cutoff,numel(wL_GMV));
wL_top_GMV = wL_GMV(1:kL_GMV);
nL_top_GMV = nL_GMV(1:kL_GMV);
wL_other_GMV = sum(wL_GMV(kL_GMV+1:end));


short_GMV = x_GMVP < 0;
wS_GMV = x_GMVP(short_GMV);
nS_GMV = tickers(short_GMV);

[wS_GMV, indexS_GMV] = sort(wS_GMV, 'ascend');
nS_GMV = nS_GMV(indexS_GMV);

kS_GMV = min(cutoff,numel(wS_GMV));
wS_top_GMV = wS_GMV(1:kS_GMV);
nS_top_GMV = nS_GMV(1:kS_GMV);
wS_other_GMV = sum(wS_GMV(kS_GMV+1:end));


%% RISK-FREE RATE AND CAPITAL MARKET LINE


Rf = (1 + Rf_mean).^frequency - 1;


A_bar = 1/sqrt(a+c*Rf^2-2*Rf*b);
z = V \ (mu - Rf*e);

x_tangency = z / (e' * z);
m_tangency = mu' * x_tangency;
sigma2_tangency = x_tangency' * V * x_tangency;
sigma_tangency = sqrt(sigma2_tangency);

sigma_cml = A_bar*(m-Rf);
m_cml = Rf + ( (m_tangency - Rf) / sigma_tangency ) * sigma_cml;


sig_i = sqrt(diag(V));


%% TANGENCY PORTFOLIO COMPOSITION


long_TAN = x_tangency > 0;
wL_TAN = x_tangency(long_TAN);
nL_TAN = tickers(long_TAN);

[wL_TAN, indexL_TAN] = sort(wL_TAN, 'descend');
nL_TAN = nL_TAN(indexL_TAN);

kL_TAN = min(cutoff,numel(wL_TAN));
wL_top_TAN = wL_TAN(1:kL_TAN);
nL_top_TAN = nL_TAN(1:kL_TAN);
wL_other_TAN = sum(wL_TAN(kL_TAN+1:end));


short_TAN = x_tangency < 0;
wS_TAN = x_tangency(short_TAN);
nS_TAN = tickers(short_TAN);

[wS_TAN, indexS_TAN] = sort(wS_TAN, 'ascend');
nS_TAN = nS_TAN(indexS_TAN);

kS_TAN = min(cutoff,numel(wS_TAN));
wS_top_TAN = wS_TAN(1:kS_TAN);
nS_top_TAN = nS_TAN(1:kS_TAN);
wS_other_TAN = sum(wS_TAN(kS_TAN+1:end));


%% SHARPE RATIOS


sharpe_tan = (m_tangency - Rf) / sigma_tangency;
sharpe_gmv = (m_GMVP - Rf) / sigma_GMVP;

fprintf('Tangency Sharpe: %.3f | GMV Sharpe: %.3f\n', sharpe_tan, sharpe_gmv);


%% TABLES EXPORT


if export_flag
    if ~exist('outputs', 'dir')
        mkdir('outputs')
    end

    if ~exist('outputs/figures', 'dir')
        mkdir('outputs/figures')
    end

    if ~exist('outputs/tables', 'dir')
        mkdir('outputs/tables')
    end
end


FF3_Regression_Table = table(string(tickers(:)), alpha, betas(:,1), betas(:,2), betas(:,3), R2, ...
    'VariableNames', {'Asset','Alpha_Daily','Beta_Mkt','Beta_SMB','Beta_HML','R2'});

Mu_Table = table(string(tickers(:)), mu, 'VariableNames', {'Asset','Mu_Annual'});

Weights_GMVP = table(string(tickers(:)), x_GMVP, 'VariableNames', {'Asset','Weight'});
Weights_Tangency = table(string(tickers(:)), x_tangency, 'VariableNames', {'Asset','Weight'});

Frontier_Table = table(sigma(:), m(:), 'VariableNames', {'Sigma','Mu'});
CML_Table = table(sigma_cml(:), m_cml(:), 'VariableNames', {'Sigma','Mu'});


if export_flag
    writetable(FF3_Regression_Table,'outputs/tables/ff3_regression_table.csv');
    writetable(Mu_Table,'outputs/tables/mu.csv');
    writetable(Weights_GMVP,'outputs/tables/weights_gmv.csv');
    writetable(Weights_Tangency,'outputs/tables/weights_tangency.csv');
    writetable(Frontier_Table,'outputs/tables/frontier_points.csv');
    writetable(CML_Table,'outputs/tables/cml_points.csv');
end


%% PLOTS


% === Plot 1: Efficient frontier, no risk-free ===


figure;
hold on; box on; grid on;

plot(sigma, m, 'b-', 'LineWidth', 1.5);
plot(sigma_GMVP, m_GMVP, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');

xlim([0 0.45]);
xticks(0:0.05:0.45);
ylim([0 0.5]);
xlabel('Annual volatility (\sigma)','FontSize',12);
ylabel('Annual expected return (\mu)','FontSize',12);
title('Mean–Variance Efficient Frontier','FontSize',14);

legend('Efficient Frontier','Global Min-Var Portfolio','Location','southeast');
set(gca,'FontSize',11);

if export_flag
    exportgraphics(gcf, 'outputs/figures/1_frontier.png', 'Resolution', 300);
end

close;


% === Plot 2: Efficient frontier, no risk-free, with individual assets ===


figure;
hold on; box on; grid on;

plot(sigma, m, 'b-', 'LineWidth', 1.5);
plot(sigma_GMVP, m_GMVP, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
plot(sig_i, mu, 'o', 'MarkerSize', 4, 'MarkerFaceColor','#B1CFFC','MarkerEdgeColor','#B1CFFC');

xlim([0 0.45]);
xticks(0:0.05:0.45);
ylim([0 0.5]);
xlabel('Annual volatility (\sigma)','FontSize',12);
ylabel('Annual expected return (\mu)','FontSize',12);
title('Comparison with Individual Assets','FontSize',14);

legend('Efficient Frontier','Global Min-Var Portfolio','Individual Assets','Location','southeast');
set(gca,'FontSize',11);

if export_flag
    exportgraphics(gcf, 'outputs/figures/2_frontier_assets.png', 'Resolution', 300);
end

close;


% === Plot 3: Efficient frontier + CML and tangency ===


figure;
hold on; box on; grid on;

plot(sigma, m, 'b-', 'LineWidth', 1.5);
plot(sigma_cml, m_cml, 'k-', 'LineWidth', 1.5);

plot(sigma_GMVP, m_GMVP, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
plot(sigma_tangency, m_tangency, 'go', 'MarkerSize', 4, 'MarkerFaceColor','g');

xlim([0 0.45]);
xticks(0:0.05:0.45);
ylim([0 0.5]);
xlabel('Annual volatility (\sigma)','FontSize',12);
ylabel('Annual expected return (\mu)','FontSize',12);
title('Efficient Frontier (CML) with risk-free asset','FontSize',14);

legend('Efficient Frontier','CML','Global Min-Var Portfolio','Tangency Portfolio','Location','southeast');

if export_flag
    exportgraphics(gcf, 'outputs/figures/3_frontier_riskfree.png', 'Resolution', 300);
end

close;


% === Plot 4: Efficient frontier + CML and tangency, with individual assets ===


figure;
hold on; box on; grid on;

plot(sigma, m, 'b-', 'LineWidth', 1.5);
plot(sigma_cml, m_cml, 'k-', 'LineWidth', 1.5);

plot(sigma_GMVP, m_GMVP, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
plot(sigma_tangency, m_tangency, 'go', 'MarkerSize', 4, 'MarkerFaceColor','g');
plot(sig_i, mu, 'o', 'MarkerSize', 4, 'MarkerFaceColor','#B1CFFC','MarkerEdgeColor','#B1CFFC');

xlim([0 0.45]);
xticks(0:0.05:0.45);
ylim([0 0.5]);
xlabel('Annual volatility (\sigma)','FontSize',12);
ylabel('Annual expected return (\mu)','FontSize',12);
title('CML + Comparison with individual assets','FontSize',14);

legend('Efficient Frontier','CML','Global Min-Var Portfolio','Tangency Portfolio','Individual Assets','Location','southeast');

if export_flag
    exportgraphics(gcf, 'outputs/figures/4_frontier_riskfree_assets.png', 'Resolution', 300);
end

close;


% === Plot 5: GMV portfolio composition ===


wL_plot_GMV = [wL_top_GMV; wL_other_GMV];
nL_plot_GMV = [nL_top_GMV "Others"];

wS_plot_GMV = [wS_top_GMV; wS_other_GMV];
nS_plot_GMV = [nS_top_GMV "Others"];

xmax = 0.65;

figure('Units','normalized','Position',[0.1 0.1 0.6 0.8]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile; 
catsL_GMV = categorical(nL_plot_GMV);
catsL_GMV = reordercats(catsL_GMV, nL_plot_GMV);
b1_GMV = barh(catsL_GMV, wL_plot_GMV, 'FaceColor', '#2ECC71', 'EdgeColor', '#27AE60');
grid on; box on;
set(gca,'YDir','reverse');
xlim([0 xmax]);
xtickformat('%.0f%%');
xticks(0:0.05:xmax);
ticks = xticks;
xticklabels(compose('%.1f%%', 100*ticks));
title(append('GMV Portfolio - Long positions (Top ',string(cutoff),' + Others)'));
xlabel('Portfolio weight');
ylabel('Asset');

nexttile;
catsS_GMV = categorical(nS_plot_GMV);
catsS_GMV = reordercats(catsS_GMV, nS_plot_GMV);
b2_GMV = barh(catsS_GMV, wS_plot_GMV, 'FaceColor', '#E74C3C', 'EdgeColor', '#C0392B');
grid on; box on;
set(gca,'XDir','reverse');
set(gca,'YDir','reverse');
xlim([-xmax 0]);
xtickformat('%.0f%%');
xticks(-xmax:0.05:0);
ticks = xticks;
xticklabels(compose('%.1f%%', 100*ticks));
title(append('GMV Portfolio - Short positions (Top ',string(cutoff),' + Others)'));
xlabel('Portfolio weight');
ylabel('Asset');

if export_flag
    exportgraphics(gcf, 'outputs/figures/5_gmvp_composition.png', 'Resolution', 300);
end

close;


% === Plot 6: Tangency portfolio composition ===


wL_plot_TAN = [wL_top_TAN; wL_other_TAN];
nL_plot_TAN = [nL_top_TAN "Others"];

wS_plot_TAN = [wS_top_TAN; wS_other_TAN];
nS_plot_TAN = [nS_top_TAN "Others"];

xmax = 0.65;


figure('Units','normalized','Position',[0.1 0.1 0.6 0.8]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile; 
catsL_TAN = categorical(nL_plot_TAN);
catsL_TAN = reordercats(catsL_TAN, nL_plot_TAN);
b1_TAN = barh(catsL_TAN, wL_plot_TAN, 'FaceColor', '#2ECC71', 'EdgeColor', '#27AE60');
grid on; box on;
set(gca,'YDir','reverse');
xlim([0 xmax]);
xtickformat('%.0f%%');
xticks(0:0.05:xmax);
ticks = xticks;
xticklabels(compose('%.1f%%', 100*ticks));
title(append('Tangency Portfolio - Long positions (Top ',string(cutoff),' + Others)'));
xlabel('Portfolio weight');
ylabel('Asset');

nexttile;
catsS_TAN = categorical(nS_plot_TAN);
catsS_TAN = reordercats(catsS_TAN, nS_plot_TAN);
b2_TAN = barh(catsS_TAN, wS_plot_TAN, 'FaceColor', '#E74C3C', 'EdgeColor', '#C0392B');
grid on; box on;
set(gca,'XDir','reverse');
set(gca,'YDir','reverse');
xlim([-xmax 0]);
xtickformat('%.0f%%');
xticks(-xmax:0.05:0);
ticks = xticks;
xticklabels(compose('%.1f%%', 100*ticks));
title(append('Tangency Portfolio - Short positions (Top ',string(cutoff),' + Others)'));
xlabel('Portfolio weight');
ylabel('Asset');

if export_flag
    exportgraphics(gcf, 'outputs/figures/6_tangency_composition.png', 'Resolution', 300);
end

close;
