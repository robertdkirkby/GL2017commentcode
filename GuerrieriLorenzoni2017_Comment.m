%% Comment on Guerrieri & Lorenzoni (2017) - Credit Crises, Precautionary Savings, and the Liquidity Trap
%
% Guerrieri & Lorenzoni (2017) use a value of 2.1 for Tauchen's q (the
% hyperparameter) for the Tauchen method to approximate an AR(1) process on
% (the log of) efficiency-units-of-labour. This value of 2.1 is not mentioned
% anywhere in the paper or technical appendix despite being highly unusual
% in the literature; the Tauchen method is widely used, and typical values
% of Tauchen's q are 3 or 4.
%
% Nor is the choice of Tauchens q 2.1 directly mentioned in the codes provided by
% Guerrieri & Lorenzoni (2017) where a file (inc_process.mat) containing 
% the result of the Tauchen method approximation is instead provided.
% I discovered the value of 2.1 by reverse engineering to find the Tauchen
% q that produces (a close approximation of) the contents inc_process.mat.
%
% This is despite the crucial role played by the choice of Tauchen q=2.1.
% The rest of this comment simply creates a plot showing the stationary distribution
% of agents in general equilibrium for the Tauchen approximation based on the saved values 
% of GL2017 (inc_process.mat), and for Tauchen q=2.1, q=3, and q=4. It also
% shows an 'accurate' stationary distribution, which not only uses Tauchen
% q=4 but also increases the number of points used for the approximation (n_theta) to 51, rather than 
% the 13 used elsewhere; this illustrates what the stationary distribution
% 'should' look like if it were actually the AR(1), rather than the product
% of the Tauchen method approximation.
% A saved copy of the plot is available directly at: WEBLINK
%
% The reason this choice matters is that with q=2.1 the highest and lowest values of the (log of the)
% process on efficiency-units-of-labour is only +/-1 standard deviation. Making
% income volatility much lower than is empirically plausible. With so
% little income risk a large fraction of the population ends up very close
% to the borrowing constraint ---despite precautionary savings motives to
% move away from it--- and so the credit crisis forces savings increases on
% a substantial fraction of the population. With a more normal value of Tauchen q like 3, and more in
% line with the empirical estimates of the (log) income process to which
% the model is being calibrated, only a tiny fraction of the population
% would run the risk of being so close to their borrowing constraint and so
% the credit crisis will affect far fewer people and be relatively benign.
% Note that you can see this indirectly in Huggett (1993) Figure 2, showing
% the stationary distribution (or in the figure created by these codes);
% the decision to model labour supply introducing the alternative option of precautionary
% labour supply (instead of precautionary savings (Pijoan-Mas, 2006)), also helps the model deliver this 
% higher fraction of the population near the borrowing constraint.
% In fact, with Tauchen q=3 the zero lower bound would never have bound in
% the model and the 'New Keynesian' results would be identical to the
% 'flexible price' results (if you want to see this run the 'Example' codes with tauchen q=3).
%
% It is possible to justify the choice of q=2.1 as that which, when using
% a 12-state approximation with the Tauchen method delivers the a Markov
% process that has the same variance as the variance of the AR(1) process
% being approximated. (That is, reproduces variance(z) for z_t=rho*z_tminus1+epsilon.)
% But as described above this has the impact of removing most of the risk
% from the economy and is both an unusually low value for Tauchens q and
% key to the results of the paper. It is so unusual that it took me
% a few days to realise why I was initially completely unable to
% replicate the paper at the first attempt (I had gone with q=3, which
% tends to be the standard if not otherwise mentioned, and while I tried q=4 
% the thought of something as low as q=2 simply did not occour to me based on my 
% experience replicating many models of this kind.)
%
% The point of this comment is not so much to criticise the decision to set
% Tauchen q equal to 2.1; there are possible justifications such as that
% above about variance, or about wanting to calibrate to have a substantial fraction
% of the population near the borrowing constraint (empirically assessing
% the correct value for this fraction is very difficult).
%
% The point of this comment is that such a crucial and unusual decision was taken
% without mention or discussion in the paper. It is likely that neither the
% editor, referees, nor authors comprehended what was going on with Tauchen q as they have kept
% a robustness test for a 'high risk aversion (phi=6)' calibration in the
% paper as an important test. This is rendered pointless by Tauchen q equal
% to 2.1: what is the point of testing robustness to high risk aversion in an economy in
% which most of the risk has been removed??? 
% (The authors actually got the process from Shimer (2005) who in turn likely used the Tauchen-Hussey method, 
% rather than the Tauchen method. You should never use the Tauchen-Hussey method. It is a dreadful 
% approximation and plenty of good alternatives exist (Tauchen, Rouwenquist, Farmer-Toda, etc. For just 
% how bad Tauchen-Hussey is, see Toda (2020))
%
% PS. Other than this one issue the paper was actually a pleasure to replicate compared to
% most. It provides clear and precise descriptions of the experiments being undertaken, reports
% alternative calibrations, and the code (while only reproducing part of the paper) is clean and 
% well documented and calibrated. The paper itself is also a good and insightful read into how these
% models operate :)
% 
% Note: for this to run you need to download 'inc_process.mat' (is
% contained in codes provided by GL2017), you also need to edit line
% (roughly) 167 which tells the codes where to look to find and load 'inc_process.mat'.
% The figure itself is drawn using plotly and so you would need a copy of
% plotly installed.
%
% Note: The following code is lazily done as it is not really meant to be
% followed and read (that is purpose of the 'Example' and 'Replication' codes
% linked at WEBSITE ). It is just to produce the Figure illustrating the point of this comment
% and found at .

%% To translate Guerrieri & Lorenzoni (2017) into the standard setup of VFI Toolkit I use following:
% d variables: n_it
% aprime variables: b_it+1
% a variables: b_it
% z variables: theta_it, e_it

simoptions.parallel=2 % 4: Sparse matrix, but then put result on gpu
vfoptions.lowmemory=0

%% Set some basic variables
n_d=51 
n_a=2^10 % Guerrieri & Lorenzoni (2017) use 200 points for agent distribution; VFI is done the same but they use the EGM (endo grid method) and then use ConesaKrueger style probabilistic weights to nearest grid point for agent dist simulation
n_theta=13; % Guerrieri & Lorenzoni (2017), pg 1438, states that they use a "12 state markov chain, following the approach in Tauchen (1986)". This description can be misinterpreted as theta is in fact the combination of a two-state markov on employed/unemployed with a 12-state Tauchen approx of AR(1) applied to employment. (This is clear from their codes) [The precise wording of GL2017 is correct, just easily misread.]
n_r=1001;
Params.beta=0.9711; %Model period is one-quarter of a year.
Params.gamma=4; % Coefficient of relative risk aversion
Params.eta=1.5; % Curvature of utility from leisure
Params.psi=12.48; % Coefficient on leisure in utility
Params.pi_eu=0.0573; % Transition to unemployment (the last three comes from their code, not paper)
Params.pi_ue=0.882; % Transition to employment
Params.rho=0.967; % Persistence of productivity shock (G&L2017 call this rho)
Params.sigmasq_epsilon=0.017; % Variance of the shock to the log-AR(1) process on labour productivity.
Params.v=0.1; % Unemployment benefit
Params.B=1.6; % Bond supply
Params.Bprime=Params.B; % Bond supply is unchanging (for most of the paper); this is needed as it is part of government budget constraint that determines lump-sum tax tau_t
Params.phi=0.959; % Borrowing limit
Params.omega=0; % This is not actually needed for anything until we get to the 'Sticky Wage' model (Section 4 of GL2017, pg 1450)


%% The Tauchen Approximation itself.
% For the first example we will use the approximation actually used by
% GL2017 and stored in inc_process.mat (which was downloaded with their
% codes).
OverwriteWithGL2017Grid=1;

% None-the-less, the following walks through how to set up the Tauchen
% method approximation, and with lots of comments to explain what is going
% on. (Even though it is then overwritten, the same code without comments
% is then repeated later for the different Tauchen q values)
Params.tauchenq=2.1;
tauchenoptions.parallel=2;

% Create markov process for the exogenous income (based on idea of employment and unemployment states, following Imrohoroglu, 1989).
[theta1_grid, pi_theta1]=TauchenMethod(0, Params.sigmasq_epsilon, Params.rho, n_theta-1, Params.tauchenq,tauchenoptions);
z_grid=[0; exp(theta1_grid)];
pistar_theta1=ones(n_theta-1,1)/(n_theta-1);
for ii=1:10^4 % G&L2017, pg 1438 "when first employed, workers draw theta from its unconditional distribution"
    pistar_theta1=pi_theta1'*pistar_theta1; % There is a more efficient form to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
end

% Floden & Linde (2001) report annual values for the AR(1) on log(z) as
% rho_FL=0.9136, sigmasq_epsilon_FL=0.0426; can calculate sigmasq_z_FL=sigmasq_epsilon_FL/(1-rho_FL^2)=0.2577;
% GL2017, footnote 11 give the formulae for annual rho and sigmasqz in terms of the quarterly:
rho=Params.rho;
sigmasq_epsilon=Params.sigmasq_epsilon;
sigmasq_epsilon_annual=(1/(4^2))*(4+6*rho+4*rho^2+2*rho^3)*(sigmasq_epsilon/(1-rho^2)); % Gives 0.2572 which is very close to FL
autocovariance_annual=(1/(4^2))*(rho+2*rho^2+3*rho^3+4*rho^4+3*rho^5+2*rho^6+rho^7)*(sigmasq_epsilon/(1-rho^2));
rho_annual=autocovariance_annual/sigmasq_epsilon_annual; % Gives 0.9127, which is close to FL. Note that this depends only on quarterly rho (the quarterly sigmasq_epsilon disappears when doing the division of annual autocovariance by annual variance)
% These check out, so the values in GL2017 for rho and sigmasq_epsilon are correct.

pi_z=[(1-Params.pi_ue), Params.pi_ue*pistar_theta1'; Params.pi_eu*ones(n_theta-1,1),(1-Params.pi_eu)*pi_theta1];
% Rows were did not sum to one due to rounding errors at order of 10^(-11), fix this
pi_z=pi_z./sum(pi_z,2);
pistar_z=ones(n_theta,1)/n_theta;
for ii=1:10^4 %  % There is a more efficient way to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
    pistar_z=pi_z'*pistar_z; % Formula could be used to find stationary dist of the employment unemployment process, then just combine with stationary dist of theta1, which is already calculated
end
% "The average level of theta is chosen so that yearly output in the initial steady state is normalized to 1"
z_grid=z_grid/sum(z_grid.*pistar_z);
% Double-check that this is 1
% sum(z_grid.*pistar_z)

% That the "normalized to 1" refers to E[theta] and not E[n*theta] is clear from setting
% v=0.1 to satisfy "For the unemployment benefit, we also follow Shimer
% (2005) and set it to 40% of average labor income." (pg 1438)
% Note, they do not actually ever normalize to 1 in the codes, GL2017 has E[theta]=1.0718

% Either their reported parameter values are slightly incorrect or the
% Tauchen method code they used is not accurate, as they get slightly
% different grid, so I will just load theirs directly
if OverwriteWithGL2017Grid==1
    load ./PaperMaterials/replication-codes-for-Credit-Crises-2017-1f7cb32/inc_process.mat
    theta = [0; exp(x)];
    z_grid=theta;
    
    tol_dist = 1e-10; % tolerance lvl for distribution
    S     = length(theta);
    fin   = 0.8820;        % job-finding probability
    sep   = 0.0573;        % separation probability
    % new transition matrix
    Pr = [1-fin, fin*pr; sep*ones(S-1, 1), (1-sep)*Pr];
    % find new invariate distribution
    pr  = [0, pr];
    dif = 1;
    while dif > tol_dist
      pri = pr*Pr;
      dif = max(abs(pri-pr));
      pr  = pri;
    end
    
    pi_z=Pr;
    pistar_z=pr';
    
    % Note: GL2017 codes do not do the normalization of z_grid, they just have
    % E[theta]=1.07 rather than 1 as stated in the paper.
end

%% Grids
% Set grid for asset holdings
Params.alowerbar=-1.05; % This seems reasonable (No-one can go below -Params.phi in any case). Note that Fischer deflation experiment (second last part of paper) won't work unless this is below -1.1*phi
Params.aupperbar=14; % Not clear exactly what value is appropriate, have gone with this based on axes of Figure 1.
a_grid=(Params.aupperbar-Params.alowerbar)*(1/(exp(1)-1))*(exp(linspace(0,1,n_a)')-1)+Params.alowerbar;
r_grid=linspace(-0.05,0.05,n_r)'; % This grid is substantially wider than the actual likely equilibrium values and so is somewhat overkill.
%Bring model into the notational conventions used by the toolkit
d_grid=linspace(0,1,n_d)'; % Labor supply
p_grid=r_grid;
n_z=n_theta;
n_p=n_r;
%%
FnsToEvaluateParamNames(1).Names={};
FnsToEvaluateFn_1 = @(d_val, aprime_val,a_val,z_val) a_val; % Aggregate assets (which is this periods state)
FnsToEvaluate={FnsToEvaluateFn_1}; %, FnsToEvaluateFn_2};
GeneralEqmEqnParamNames(1).Names={'Bprime'};
GeneralEqmEqn_1 = @(AggVars,p,Bprime) AggVars(1)-Bprime; %The requirement that the aggregate assets (lending and borrowing) equal zero
GeneralEqmEqns={GeneralEqmEqn_1};
%% 
DiscountFactorParamNames={'beta'};
ReturnFn=@(d_val, aprime_val, a_val, z_val,r, gamma, psi, eta, phi, v,B,Bprime,omega) GuerrieriLorenzoni2017_ReturnFn(d_val, aprime_val, a_val, z_val,r, gamma, psi, eta, phi, v,B,Bprime,omega);
ReturnFnParamNames={'r', 'gamma', 'psi', 'eta', 'phi', 'v','B','Bprime','omega'}; %It is important that these are in same order as they appear in 'GuerrieriLorenzoni2017_ReturnFn'

%% Solve the stationary general equilibrium with inc_process.mat from GL2017.
V0=ones(n_a,n_z,'gpuArray');
%Use the toolkit to find the equilibrium price index
GEPriceParamNames={'r'}; %,'tau'

heteroagentoptions.verbose=1;
heteroagentoptions.pgrid=p_grid;

disp('Calculating price vector corresponding to the stationary eqm')
[p_eqm_initial,p_eqm_index_initial, MarketClearance_initial]=HeteroAgentStationaryEqm_Case1(V0, n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
Params.r=p_eqm_initial;
[~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames);
StationaryDist_initial_GL2017inc_process=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

% Will want this value of interest rate r to use for partial eqm comparisons later on as well.
p_eqm_initial_GL2017incprocess=p_eqm_initial;
% Note that the 'partial eqm' version in this specific case is actually just the general eqm version anyway.
StationaryDist_initial_GL2017inc_process_partialeqm=StationaryDist_initial_GL2017inc_process;

%% Solve the stationary general equilibrium with Tauchen q equal to 2.1

Params.tauchenq=2.1; % I have reverse engineered this value from the grid of GL2017 (copy of their grid is included in their codes). They themselves reverse engineered choice of roughly 2.1 so that the variance of the resulting process (variance of 12-state markov logz) is as close as possible to what it should be (variance of true AR(1) logz).

% Create markov process for the exogenous income (based on idea of employment and unemployment states, following Imrohoroglu, 1989).
[theta1_grid, pi_theta1]=TauchenMethod(0, Params.sigmasq_epsilon, Params.rho, n_theta-1, Params.tauchenq,tauchenoptions);
z_grid=[0; exp(theta1_grid)];
pistar_theta1=ones(n_theta-1,1)/(n_theta-1);
for ii=1:10^4 % G&L2017, pg 1438 "when first employed, workers draw theta from its unconditional distribution"
    pistar_theta1=pi_theta1'*pistar_theta1; % There is a more efficient form to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
end

pi_z=[(1-Params.pi_ue), Params.pi_ue*pistar_theta1'; Params.pi_eu*ones(n_theta-1,1),(1-Params.pi_eu)*pi_theta1];
% Rows were did not sum to one due to rounding errors at order of 10^(-11), fix this
pi_z=pi_z./sum(pi_z,2);
pistar_z=ones(n_theta,1)/n_theta;
for ii=1:10^4 %  % There is a more efficient way to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
    pistar_z=pi_z'*pistar_z; % Formula could be used to find stationary dist of the employment unemployment process, then just combine with stationary dist of theta1, which is already calculated
end
% "The average level of theta is chosen so that yearly output in the initial steady state is normalized to 1"
z_grid=z_grid/sum(z_grid.*pistar_z);

% % % disp('Calculating price vector corresponding to the stationary eqm')
% % % [p_eqm_initial,p_eqm_index_initial, MarketClearance_initial]=HeteroAgentStationaryEqm_Case1(V0, n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% % % Params.r=p_eqm_initial;
% % % [~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames);
% % % StationaryDist_initial_Tauchenq21=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

% Partial eqm version
Params.r=p_eqm_initial_GL2017incprocess;
[~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames);
StationaryDist_initial_Tauchenq21_partialeqm=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

%% Now switch to Tauchen q equal to 3 and recompute the Tauchen method
Params.tauchenq=3; % I have reverse engineered this value from the grid of GL2017 (copy of their grid is included in their codes). They themselves reverse engineered choice of roughly 2.1 so that the variance of the resulting process (variance of 12-state markov logz) is as close as possible to what it should be (variance of true AR(1) logz).

% Create markov process for the exogenous income (based on idea of employment and unemployment states, following Imrohoroglu, 1989).
[theta1_grid, pi_theta1]=TauchenMethod(0, Params.sigmasq_epsilon, Params.rho, n_theta-1, Params.tauchenq,tauchenoptions);
z_grid=[0; exp(theta1_grid)];
pistar_theta1=ones(n_theta-1,1)/(n_theta-1);
for ii=1:10^4 % G&L2017, pg 1438 "when first employed, workers draw theta from its unconditional distribution"
    pistar_theta1=pi_theta1'*pistar_theta1; % There is a more efficient form to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
end

pi_z=[(1-Params.pi_ue), Params.pi_ue*pistar_theta1'; Params.pi_eu*ones(n_theta-1,1),(1-Params.pi_eu)*pi_theta1];
% Rows were did not sum to one due to rounding errors at order of 10^(-11), fix this
pi_z=pi_z./sum(pi_z,2);
pistar_z=ones(n_theta,1)/n_theta;
for ii=1:10^4 %  % There is a more efficient way to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
    pistar_z=pi_z'*pistar_z; % Formula could be used to find stationary dist of the employment unemployment process, then just combine with stationary dist of theta1, which is already calculated
end
% "The average level of theta is chosen so that yearly output in the initial steady state is normalized to 1"
z_grid=z_grid/sum(z_grid.*pistar_z);

% % % disp('Calculating price vector corresponding to the stationary eqm')
% % % [p_eqm_initial,p_eqm_index_initial, MarketClearance_initial]=HeteroAgentStationaryEqm_Case1(V0, n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% % % Params.r=p_eqm_initial;
% % % [~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames);
% % % StationaryDist_initial_Tauchenq3=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

% Partial eqm version
Params.r=p_eqm_initial_GL2017incprocess;
[~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames);
StationaryDist_initial_Tauchenq3_partialeqm=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

%% Now switch to Tauchen q equal to 4 and recompute the Tauchen method
Params.tauchenq=4; % I have reverse engineered this value from the grid of GL2017 (copy of their grid is included in their codes). They themselves reverse engineered choice of roughly 2.1 so that the variance of the resulting process (variance of 12-state markov logz) is as close as possible to what it should be (variance of true AR(1) logz).

% Create markov process for the exogenous income (based on idea of employment and unemployment states, following Imrohoroglu, 1989).
[theta1_grid, pi_theta1]=TauchenMethod(0, Params.sigmasq_epsilon, Params.rho, n_theta-1, Params.tauchenq,tauchenoptions);
z_grid=[0; exp(theta1_grid)];
pistar_theta1=ones(n_theta-1,1)/(n_theta-1);
for ii=1:10^4 % G&L2017, pg 1438 "when first employed, workers draw theta from its unconditional distribution"
    pistar_theta1=pi_theta1'*pistar_theta1; % There is a more efficient form to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
end

pi_z=[(1-Params.pi_ue), Params.pi_ue*pistar_theta1'; Params.pi_eu*ones(n_theta-1,1),(1-Params.pi_eu)*pi_theta1];
% Rows were did not sum to one due to rounding errors at order of 10^(-11), fix this
pi_z=pi_z./sum(pi_z,2);
pistar_z=ones(n_theta,1)/n_theta;
for ii=1:10^4 %  % There is a more efficient way to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
    pistar_z=pi_z'*pistar_z; % Formula could be used to find stationary dist of the employment unemployment process, then just combine with stationary dist of theta1, which is already calculated
end
% "The average level of theta is chosen so that yearly output in the initial steady state is normalized to 1"
z_grid=z_grid/sum(z_grid.*pistar_z);

% % % disp('Calculating price vector corresponding to the stationary eqm')
% % % [p_eqm_initial,p_eqm_index_initial, MarketClearance_initial]=HeteroAgentStationaryEqm_Case1(V0, n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% % % Params.r=p_eqm_initial;
% % % [~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames);
% % % StationaryDist_initial_Tauchenq4=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

% Partial eqm version
Params.r=p_eqm_initial_GL2017incprocess;
[~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames);
StationaryDist_initial_Tauchenq4_partialeqm=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

%% Now switch to 'accurate' approximation Tauchen q equal to 4 and n_theta equal to 51
n_theta=51;
Params.tauchenq=4; % I have reverse engineered this value from the grid of GL2017 (copy of their grid is included in their codes). They themselves reverse engineered choice of roughly 2.1 so that the variance of the resulting process (variance of 12-state markov logz) is as close as possible to what it should be (variance of true AR(1) logz).

n_z=n_theta;
V0=ones(n_a,n_z,'gpuArray');

vfoptions.lowmemory=1;
simoptions.parallel=4; % >2 solves using sparse matrices on cpu (4 means switch output back to gpu) [sparse matrices allow for bigger grid sizes without running out of memory]

% Create markov process for the exogenous income (based on idea of employment and unemployment states, following Imrohoroglu, 1989).
[theta1_grid, pi_theta1]=TauchenMethod(0, Params.sigmasq_epsilon, Params.rho, n_theta-1, Params.tauchenq,tauchenoptions);
z_grid=[0; exp(theta1_grid)];
pistar_theta1=ones(n_theta-1,1)/(n_theta-1);
for ii=1:10^4 % G&L2017, pg 1438 "when first employed, workers draw theta from its unconditional distribution"
    pistar_theta1=pi_theta1'*pistar_theta1; % There is a more efficient form to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
end

pi_z=[(1-Params.pi_ue), Params.pi_ue*pistar_theta1'; Params.pi_eu*ones(n_theta-1,1),(1-Params.pi_eu)*pi_theta1];
% Rows were did not sum to one due to rounding errors at order of 10^(-11), fix this
pi_z=pi_z./sum(pi_z,2);
pistar_z=ones(n_theta,1)/n_theta;
for ii=1:10^4 %  % There is a more efficient way to do this directly from a formula but I am feeling lazy. %FIX THIS LATER!!!
    pistar_z=pi_z'*pistar_z; % Formula could be used to find stationary dist of the employment unemployment process, then just combine with stationary dist of theta1, which is already calculated
end
% "The average level of theta is chosen so that yearly output in the initial steady state is normalized to 1"
z_grid=z_grid/sum(z_grid.*pistar_z);

% Floden & Linde (2001) report annual values for the AR(1) on log(z) as
% rho_FL=0.9136, sigmasq_epsilon_FL=0.0426; can calculate sigmasq_z_FL=sigmasq_epsilon_FL/(1-rho_FL^2)=0.2577;
% GL2017, footnote 11 give the formulae for annual rho and sigmasqz in terms of the quarterly:
rho=Params.rho;
sigmasq_epsilon=Params.sigmasq_epsilon;
sigmasq_epsilon_annual=(1/(4^2))*(4+6*rho+4*rho^2+2*rho^3)*(sigmasq_epsilon/(1-rho^2)); % Gives 0.2572 which is very close to FL
autocovariance_annual=(1/(4^2))*(rho+2*rho^2+3*rho^3+4*rho^4+3*rho^5+2*rho^6+rho^7)*(sigmasq_epsilon/(1-rho^2));
rho_annual=autocovariance_annual/sigmasq_epsilon_annual; % Gives 0.9127, which is close to FL. Note that this depends only on quarterly rho (the quarterly sigmasq_epsilon disappears when doing the division of annual autocovariance by annual variance)
% These check out, so the values in GL2017 for rho and sigmasq_epsilon are correct.

% % % disp('Calculating price vector corresponding to the stationary eqm')
% % % [p_eqm_initial,p_eqm_index_initial, MarketClearance_initial]=HeteroAgentStationaryEqm_Case1(V0, n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% % % Params.r=p_eqm_initial;
% % % [~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
% % % StationaryDist_initial_accurate=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

% Partial eqm version
Params.r=p_eqm_initial_GL2017incprocess;
[~,Policy_initial]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
StationaryDist_initial_accurate_partialeqm=StationaryDist_Case1(Policy_initial,n_d,n_a,n_z,pi_z, simoptions);

%%
save ./SavedOutput/GL2017comment.mat StationaryDist_initial_GL2017inc_process StationaryDist_initial_Tauchenq21 StationaryDist_initial_Tauchenq3 StationaryDist_initial_Tauchenq4 StationaryDist_initial_accurate

save ./SavedOutput/GL2017comment_partialeqm.mat StationaryDist_initial_GL2017inc_process_partialeqm StationaryDist_initial_Tauchenq21_partialeqm StationaryDist_initial_Tauchenq3_partialeqm StationaryDist_initial_Tauchenq4_partialeqm StationaryDist_initial_accurate_partialeqm

%% Draw graph of these stationary distributions allowing for the general equilibrium effects.
% I use plotly to draw this.

% Plot pdf
trace1= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_GL2017inc_process,2)),'name', 'GL2017 inc_process','type', 'scatter','marker',struct('size',1));
trace2= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_Tauchenq21,2)),'name', 'Tauchen q=2.1','type', 'scatter','marker',struct('size',1));
trace3= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_Tauchenq3,2)),'name', 'Tauchen q=3','type', 'scatter','marker',struct('size',1));
trace4= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_Tauchenq4,2)),'name', 'Tauchen q=4','type', 'scatter','marker',struct('size',1));
trace5= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_accurate,2)),'name', 'Accurate (q=4, n_theta=51)','type', 'scatter','marker',struct('size',1));
data = {trace1,trace2,trace3,trace4,trace5};
layout = struct('title', 'Agent Stationary Distribution for model of Guerrieri & Lorenzoni (2017)','showlegend', true,'width', 800,...
    'xaxis', struct('domain', [0, 0.9],'title','Asset holdings (a)'),...
    'yaxis', struct('title', 'Probability Distribution Fn','titlefont', struct('color', 'black'),'tickfont', struct('color', 'black'),'anchor', 'free','side', 'left','position',0) );
response = plotly(data, struct('layout', layout, 'filename', 'GL2017commentFigure', 'fileopt', 'overwrite'));
response.data=data; response.layout=layout;
saveplotlyfig(response, 'GL2017commentFigure.pdf')

% Plot cdf
trace1= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_GL2017inc_process,2))),'name', 'GL2017 inc_process','type', 'scatter','marker',struct('size',1));
trace2= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_Tauchenq21,2))),'name', 'Tauchen q=2.1','type', 'scatter','marker',struct('size',1));
trace3= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_Tauchenq3,2))),'name', 'Tauchen q=3','type', 'scatter','marker',struct('size',1));
trace4= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_Tauchenq4,2))),'name', 'Tauchen q=4','type', 'scatter','marker',struct('size',1));
trace5= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_accurate,2))),'name', 'Accurate (q=4, n_theta=51)','type', 'scatter','marker',struct('size',1));
data = {trace1,trace2,trace3,trace4,trace5};
layout = struct('title', 'Agent Stationary Distribution for model of Guerrieri & Lorenzoni (2017)','showlegend', true,'width', 800,...
    'xaxis', struct('domain', [0, 0.9],'title','Asset holdings (a)'),...
    'yaxis', struct('title', 'Cumulative Distribution Fn','titlefont', struct('color', 'black'),'tickfont', struct('color', 'black'),'anchor', 'free','side', 'left','position',0) );
response = plotly(data, struct('layout', layout, 'filename', 'GL2017commentFigure_cdf', 'fileopt', 'overwrite'));
response.data=data; response.layout=layout;
saveplotlyfig(response, 'GL2017commentFigure_cdf.pdf')

%% Draw graph of these stationary distributions if all of them used the same interest rate r (namely that which is general eqm when using GL2017 income process).
% I use plotly to draw this.


% Plot pdf
trace1= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_GL2017inc_process_partialeqm,2)),'name', 'GL2017 inc_process','type', 'scatter','marker',struct('size',1));
trace2= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_Tauchenq21_partialeqm,2)),'name', 'Tauchen q=2.1','type', 'scatter','marker',struct('size',1));
trace3= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_Tauchenq3_partialeqm,2)),'name', 'Tauchen q=3','type', 'scatter','marker',struct('size',1));
trace4= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_Tauchenq4_partialeqm,2)),'name', 'Tauchen q=4','type', 'scatter','marker',struct('size',1));
trace5= struct('x', a_grid,'y',gather(sum(StationaryDist_initial_accurate_partialeqm,2)),'name', 'Accurate (q=4, n_theta=51)','type', 'scatter','marker',struct('size',1));
data = {trace1,trace2,trace3,trace4,trace5};
layout = struct('title', 'Partial Eqm Agent Stationary Distribution for model of Guerrieri & Lorenzoni (2017)','showlegend', true,'width', 800,...
    'xaxis', struct('domain', [0, 0.9],'title','Asset holdings (a)'),...
    'yaxis', struct('title', 'Probability Distribution Fn','titlefont', struct('color', 'black'),'tickfont', struct('color', 'black'),'anchor', 'free','side', 'left','position',0) );
response = plotly(data, struct('layout', layout, 'filename', 'GL2017commentFigure_partialeqm', 'fileopt', 'overwrite'));
response.data=data; response.layout=layout;
saveplotlyfig(response, 'GL2017commentFigure_partialeqm.pdf')

% Plot cdf
trace1= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_GL2017inc_process_partialeqm,2))),'name', 'GL2017 inc_process','type', 'scatter','marker',struct('size',1));
trace2= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_Tauchenq21_partialeqm,2))),'name', 'Tauchen q=2.1','type', 'scatter','marker',struct('size',1));
trace3= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_Tauchenq3_partialeqm,2))),'name', 'Tauchen q=3','type', 'scatter','marker',struct('size',1));
trace4= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_Tauchenq4_partialeqm,2))),'name', 'Tauchen q=4','type', 'scatter','marker',struct('size',1));
trace5= struct('x', a_grid,'y',gather(cumsum(sum(StationaryDist_initial_accurate_partialeqm,2))),'name', 'Accurate (q=4, n_theta=51)','type', 'scatter','marker',struct('size',1));
data = {trace1,trace2,trace3,trace4,trace5};
layout = struct('title', 'Partial Eqm Agent Stationary Distribution for model of Guerrieri & Lorenzoni (2017)','showlegend', true,'width', 800,...
    'xaxis', struct('domain', [0, 0.9],'title','Asset holdings (a)'),...
    'yaxis', struct('title', 'Cumulative Distribution Fn','titlefont', struct('color', 'black'),'tickfont', struct('color', 'black'),'anchor', 'free','side', 'left','position',0) );
response = plotly(data, struct('layout', layout, 'filename', 'GL2017commentFigure_cdf_partialeqm', 'fileopt', 'overwrite'));
response.data=data; response.layout=layout;
saveplotlyfig(response, 'GL2017commentFigure_cdf_partialeqm.pdf')

%% Extra: Takes a closer look at the inc_process.mat from the Guerrieri & Lorenzoni (2017) codes.
% This is how I figured out that they use roughly tauchenq=2.1, and that this is
% based on targeting that sigmasqz takes the correct value.
% (repeatedly ran following, manually changing the value of Tauchen q).

% Load the log-income process (which is 12-state Tauchen method approximation of AR(1)) (already done above)
% load ./PaperMaterials/replication-codes-for-Credit-Crises-2017-1f7cb32/inc_process.mat

% Simulate a time series
burnin=10^3;
T=10^4
currenttimeseries_index=1
cumPr=cumsum(Pr,2);
% Start with the indexes
for ii=1:burnin
    [~,currenttimeseries_index]=max(cumPr(currenttimeseries_index,:)>rand(1,1));
end
for ii=1:T
    [~,currenttimeseries_index]=max(cumPr(currenttimeseries_index,:)>rand(1,1));
    timeseries_index(ii)=currenttimeseries_index;
end
% Now the values
timeseries_values=x(timeseries_index);
% Variance of these
var(timeseries_values)

% It should be
rho=0.967;
sigmasq_epsilon=0.017;
sigmasq_z=sigmasq_epsilon/(1-rho^2)


% Use Tauchen Method
tauchenoptions.parallel=1;
Params.tauchenq=2.1
n_theta=13;
Params.rho=rho;
Params.sigmasq_epsilon=sigmasq_epsilon;
[theta1_grid, pi_theta1]=TauchenMethod(0, Params.sigmasq_epsilon, Params.rho, n_theta-1, Params.tauchenq,tauchenoptions);

cumpi_theta1=cumsum(pi_theta1,2);
% Start with the indexes
for ii=1:burnin
    [~,currenttimeseries_index]=max(cumpi_theta1(currenttimeseries_index,:)>rand(1,1));
end
for ii=1:T
    [~,currenttimeseries_index]=max(cumpi_theta1(currenttimeseries_index,:)>rand(1,1));
    timeseries_index(ii)=currenttimeseries_index;
end
% Now the values
timeseries_values=theta1_grid(timeseries_index);
% Variance of these
var(timeseries_values)

