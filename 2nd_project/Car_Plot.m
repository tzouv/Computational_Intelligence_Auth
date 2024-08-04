% This script plots the car movement (x,y) based on the simulation results,
% at the same graph with the obstacle

% Author: Tzouvaras Evangelos

% Define X and Y coordinates of the obstacle points
X_obstacle = [5, 5, 6, 6, 7, 7, 10];
Y_obstacle = [0, 1, 1, 2, 2, 3, 3];
figure;

% Plot the obstacle
plot(X_obstacle, Y_obstacle, 'LineWidth', 2);

% Set axis labels and title and axis limits
xlabel('X-axis');   
ylabel('Y-axis');
title('Car Movement');
xlim([0, 12]);
ylim([0, 4]);

hold on;

% Desired point coordinates
X_des = 10;
Y_des = 3.2;

% Plot the Desired Position point
plot(X_des, Y_des, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 5);

% Plot the (X-Y) car route
plot(out.X.Data, out.Y.Data);


