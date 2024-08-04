% This script produces the graphs for the system response based on simulink
% data

% Author: Tzouvaras Evangelos

% Plot refence input and output
figure(1);
plot(out.r.Time, out.r.Data);
hold on;
plot(out.y.Time, out.y.Data);
title('System Response: ke = 5, ki = 0.5, k = 20');
xlabel('Time(sec)');
ylabel('Satellite Angle(degrees)');
ylim([0 100]);

% Calculate the rise time of the signal
rise = risetime(out.y.Data, out.y.Time);
fprintf("\n\nRise Time(sec): %d", rise);

% Calculate the overshoot factor
os = overshoot(out.y.Data, out.y.Time);
os = (max(out.y.Data)-out.y.Data(end,:))/out.y.Data(end,:)*100;
fprintf("\nOvershoot factor(%%): %d", os);

% % Plot the control signal
% figure(2);
% plot(out.r.Time, out.r.Data);
% hold on;
% plot(out.u.Time, out.u.Data);
% title('System Response: ke = ki = 5, k = 100');
% xlabel('Time(sec)');
% ylabel('Control Signal');
% ylim([0 300]);

