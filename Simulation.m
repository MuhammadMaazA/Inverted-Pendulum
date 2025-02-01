% Parameters
m = 0.8;  % Mass of pendulum (kg)
M = 0.5;  % Mass of cart (kg)
L = 0.3;  % Length of pendulum (m)
g = 9.81; % Gravity (m/s^2)
d = 0;  % Damping coefficient (N·s/m)

% State-space model
A = [0, 1, 0, 0;
     0, -d/M, -(m*g)/M, 0;
     0, 0, 0, 1;
     0, d/(M*L), (M+m)*g/(M*L), 0];
B = [0; 1/M; 0; -1/(M*L)];
C = [1, 0, 0, 0];
D = 0;

% Design LQR Controller
Q = diag([100, 1, 10, 1]); % State cost weights
R = 0.1;                   % Control effort cost
K = lqr(A, B, Q, R);       % LQR gain

% Closed-loop system
A_cl = A - B*K;
sys_cl = ss(A_cl, B, C, D);

% Initial conditions (fully stabilized)
x0 = [0; 0; 0; 0]; % Initial state: [x, x_dot, theta, theta_dot]

% Time vector
t = 0:0.01:10;

% External disturbance force
disturbance = zeros(size(t));      % No disturbance initially
t_d = 2;                           % Time of disturbance (seconds)
force_magnitude = 40;              % Magnitude of disturbance force (N)
disturbance(t > t_d & t < t_d+0.1) = force_magnitude; % Apply disturbance for 0.1s

% Simulate the system
[Y, T, X] = lsim(sys_cl, disturbance, t, x0);

% Plot the results
figure;
subplot(2,1,1);
plot(T, X(:,1), 'LineWidth', 1.5);
ylabel('Cart Position (m)');
title('Response to External Disturbance');
grid on;

subplot(2,1,2);
plot(T, X(:,3) * 180/pi, 'LineWidth', 1.5); % Convert rad to degrees
ylabel('Pendulum Angle (degrees)');
xlabel('Time (s)');
grid on;

% Animation setup
cart_width = 0.4;
cart_height = 0.2;
pendulum_length = L;

figure;
axis([-1 1 -0.5 1]); % Set axis limits
hold on;
grid on;
xlabel('Cart Position (m)');
ylabel('Pendulum Height (m)');
title('Cart-Pendulum Stabilization Animation with Disturbance');

% Draw the cart
cart = rectangle('Position', [0, 0, cart_width, cart_height], 'FaceColor', [0 0.5 0.5]);

% Draw the pendulum
pendulum = line([0, 0], [0, 0], 'LineWidth', 2, 'Color', 'r');

% Simulation loop for animation
for i = 1:length(T)
    % Update cart position
    cart_x = X(i, 1) - cart_width / 2; % Center cart at X(i, 1)
    set(cart, 'Position', [cart_x, 0, cart_width, cart_height]);

    % Update pendulum position
    pendulum_x = X(i, 1) + pendulum_length * sin(X(i, 3));
    pendulum_y = cart_height + pendulum_length * cos(X(i, 3));
    set(pendulum, 'XData', [X(i, 1), pendulum_x]);
    set(pendulum, 'YData', [cart_height, pendulum_y]);

    % Pause to control animation speed
    pause(0.01);
end