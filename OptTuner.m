

%% Part 2
clear all
close all

load COVID_STL.mat

figure()
A = [0.7 0.04 0 0; 0.05 0.85 0 0; 0 0.1 1 0; 0 0.01 0 1];
%A1 = [0.7 0.5 0 0; 0.05 0.39 0 0; 0 0.1 1 0; 0 0.01 0 1]; % change (1,2) from 0.04 to 0.5. change 2,2 from 0.85 to 0.39.
A1 = [0.7 0.04 0 0; 0.05 0.85 0 0; 0 0.1 1 0; 0 0.01 0 1]
B = zeros(4,1); %B = zeros, so Bu(t) = 0, which means there are no external inputs. state x(t) only evolves using A.

x0 = [0.9 0.1 0 0]; %initial state of SIRD

C = eye(4);
D = zeros(4,1);

%sys_sir_base = ss(A,B,eye(4),zeros(4,1),1); %OG code

sys_sir_base = ss(A1,B,C,D,1); 

%A = initial state, B  = zeros, so x(t) only evolves with A.
% C is a 4x4 identity matrix, and D is zero, which means that y(t), the output
%of the system is just the current state of x(t)
%the last ",1" means sample time 1s

Y = lsim(sys_sir_base, zeros(1000,1), linspace(0,999,1000),  x0); % u = zeros, meaning there are no external inputs at time t

plot(Y); % plot the output trajectory

legend('S','I','R','D');
xlabel('Time')
ylabel('Percentage Population');

%% new_cases and new_deaths shows us the amount of NEW cases/deaths per day (original data tables gave us cumulative values)

POP_STL = 1290497;

for i = 1:157
    new_cases(i) =  cases_STL(i+1)-cases_STL(i);
    infected_percent(i) = cases_STL(i)/POP_STL;
end 

for i = 1:157
    new_deaths(i) = deaths_STL(i+1)-deaths_STL(i);
    deaths_percent(i) = deaths_STL(i)/POP_STL;
end

hold on
plot(infected_percent)
hold on
plot(deaths_percent)
legend('S','I','R','D','I data','D data')

xlim([0,160])
ylim([0,0.4])



%%

S = [0.7 0.04 0 0; 0.05 0.85 0 0; 0 0.1 1 0; 0 0.01 0 1]; % The same matrix as A

Q = @(x) [x(1) x(2) 0 0; x(3) x(4) 0 0; 0 x(5) 1 0; 0 x(6) 0 1]; % Coefficient Matrix, but with variables for entries we can actually change 
x1 = [S(1,1) S(1,2) S(2,1) S(2,2) S(3,2) S(4,2)]; % Extracts the entries of Matrix A that we can actually change

y = infected_percent;

ObjMat = @(Q) Q*x0'; % Data pt 2, given x0 as Data pt 1

ObjFunI = @(ObjMat) (ObjMat(2,1)-y(1,2))*(ObjMat(2,1)-y(1,2))'; % Least Squares Function for Infected
ObjFunD = @(ObjMat) (ObjMat(4,1)-y(1,2))*(ObjMat(4,1)-y(1,2))'; % Least Squares Function for Deceased

ObjFunctionI = @(x) feval(ObjFunI, feval(ObjMat, feval(Q, x))); % The same function as ObjFunI & D, just with x as the input variable (vector)

ConEq = [1 0 1 0 0 0; 0 1 0 1 1 1]; % Entries in any column Sum to 1
bEq = [1; 1]; % Entries in any Column Sum to 1

lb = zeros(6,1); % All entries non-negative
ub = ones(6,1); % All entries less than or equal to 1

A12vectI = fmincon(ObjFunctionI, x1, [], [], ConEq, bEq, lb, ub); %Vector of best entries

A12MatI = [A12vectI(1) A12vectI(2) 0 0; A12vectI(3) A12vectI(4) 0 0; 0 A12vectI(5) 1 0; 0 A12vectI(6) 0 1] % Matrix representation of results


