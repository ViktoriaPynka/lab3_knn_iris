%% Лабораторна робота №3
% Тема: Дослідження побудови класифікатора KNN

clc;
clear;
close all;

%% 1. Завантаження бази даних Iris
data = readtable('IrisData_full.csv');

% Виведення перших 10 записів
disp('Перші 10 записів бази даних:');
disp(head(data, 10));

% Визначення розміру бази даних
disp('Розмір бази даних:');
disp(size(data));




%% 3. Нормалізація параметрів квітів ірису

% Виділяємо 4 числові характеристики квітів
X = data_shuffled{:, 1:4};

% Знаходимо мінімальні та максимальні значення
X_min = min(X);
X_max = max(X);

% Виконуємо Min-Max нормалізацію
X_norm = (X - X_min) ./ (X_max - X_min);

% Виводимо перші 10 нормалізованих записів
disp('Перші 10 записів після нормалізації:');
disp(X_norm(1:10, :));


%% 3.1. Візуалізація даних до та після нормалізації

% Визначаємо види квітів
species = categorical(data_shuffled{:, 5});

% Графік 1. Дані до нормалізації
figure;
gscatter(X(:, 1), X(:, 3), species);
title('Розподіл ірисів до нормалізації');
xlabel('Довжина чашолистка (см)');
ylabel('Довжина пелюстки (см)');
legend('Location', 'best');
grid on;

% Графік 2. Дані після нормалізації
figure;
gscatter(X_norm(:, 1), X_norm(:, 3), species);
title('Розподіл ірисів після нормалізації');
xlabel('Нормалізована довжина чашолистка');
ylabel('Нормалізована довжина пелюстки');
legend('Location', 'best');
grid on;


%% 4. Розподіл даних на навчальну та тестову вибірки

% Визначаємо кількість записів
n = size(X, 1);

% 80% даних використовуємо для навчання
n_train = round(0.8 * n);

% Розділяємо числові характеристики
X_train = X(1:n_train, :);
X_test = X(n_train+1:end, :);

% Розділяємо назви видів квітів
Y_train = species(1:n_train);
Y_test = species(n_train+1:end);

% Нормалізуємо за параметрами навчальної вибірки
train_min = min(X_train);
train_max = max(X_train);
train_range = train_max - train_min;
train_range(train_range == 0) = 1;

X_train_norm = (X_train - train_min) ./ train_range;
X_test_norm = (X_test - train_min) ./ train_range;

% Виводимо результати
disp('Кількість записів навчальної вибірки:');
disp(size(X_train, 1));

disp('Кількість записів тестової вибірки:');
disp(size(X_test, 1));


%% 5. Навчання KNN з різними значеннями K

% Значення K, які будемо перевіряти
K_values = 1:20;

% Масив для збереження точності
accuracy = zeros(length(K_values), 1);

% Навчаємо та перевіряємо моделі
for i = 1:length(K_values)

    k = K_values(i);

    % Створюємо KNN-класифікатор
    model = fitcknn(X_train_norm, Y_train, ...
        'NumNeighbors', k);

    % Прогнозуємо види квітів тестової вибірки
    Y_pred = predict(model, X_test_norm);

    % Обчислюємо точність класифікації
    accuracy(i) = mean(Y_pred == Y_test) * 100;

    % Виводимо результат
    fprintf('K = %d, точність = %.2f%%\n', ...
        k, accuracy(i));

end


%% 5.1. Стовпчикова діаграма точності KNN

% Знаходимо найкраще значення K
[best_accuracy, best_idx] = max(accuracy);
best_K = K_values(best_idx);

% Створюємо графік
figure;

b = bar(K_values, accuracy, 0.7);
b.FaceColor = 'flat';

% Задаємо колір стовпчиків
b.CData = repmat([0.2 0.6 0.9], length(K_values), 1);

% Виділяємо найкраще значення червоним
b.CData(best_idx, :) = [0.95 0.3 0.3];

% Оформлення графіка
title('Порівняння точності KNN для різних K');
xlabel('Кількість найближчих сусідів (K)');
ylabel('Точність класифікації (%)');

xticks(K_values);
ylim([0 105]);
grid on;

% Підписуємо найкращий результат
text(best_K, best_accuracy + 2, ...
    sprintf('%.2f%%', best_accuracy), ...
    'HorizontalAlignment', 'center', ...
    'FontWeight', 'bold');

% Виводимо результат
fprintf('\nНайкраще значення K = %d\n', best_K);
fprintf('Максимальна точність = %.2f%%\n', best_accuracy);



%% 6. Візуалізація меж класифікації KNN

% Використовуємо дві характеристики
X_2D = X_train_norm(:, [1 3]);

% Створюємо модель для K = 1
model_2D = fitcknn(X_2D, Y_train, ...
    'NumNeighbors', 1);

% Створюємо сітку точок
[x1, x2] = meshgrid( ...
    linspace(-0.05, 1.05, 250), ...
    linspace(-0.05, 1.05, 250));

grid_points = [x1(:), x2(:)];

% Прогнозуємо класи
predicted = predict(model_2D, grid_points);

% Перетворюємо класи на числа
class_names = categories(Y_train);
[~, class_idx] = ismember(cellstr(predicted), class_names);
class_grid = reshape(class_idx, size(x1));

% Створюємо графік
figure;

% Кольорові області для трьох класів
contourf(x1, x2, class_grid, ...
    [0.5 1.5 2.5 3.5], ...
    'LineColor', 'none');

hold on;

% Світлі кольори областей
colormap([1.0 0.75 0.75;
    0.75 1.0 0.75;
    0.75 0.80 1.0]);

caxis([0.5 3.5]);

% Точки кожного виду ірисів
colors = [0.9 0.2 0.2;
    0.1 0.6 0.2;
    0.2 0.3 0.9];

% Відображаємо точки кожного виду
h = gobjects(3, 1);

for i = 1:3
    idx = Y_train == class_names{i};

    h(i) = scatter(X_2D(idx,1), X_2D(idx,2), ...
        40, colors(i,:), 'filled', ...
        'MarkerEdgeColor', 'k');
end

% Оформлення
title('Межі класифікації KNN при K = 1');
xlabel('Нормалізована довжина чашолистка');
ylabel('Нормалізована довжина пелюстки');

legend(h, class_names, 'Location', 'northeast');

xlim([-0.05 1.05]);
ylim([-0.05 1.05]);

grid on;
hold off;


%% 6.1. Порівняння меж класифікації для різних K

% Значення K для порівняння
K_compare = [3 6 10];

for j = 1:length(K_compare)

    k = K_compare(j);

    % Навчаємо KNN за двома характеристиками
    model_2D = fitcknn(X_2D, Y_train, ...
        'NumNeighbors', k);

    % Прогнозуємо класи для точок сітки
    predicted = predict(model_2D, grid_points);

    % Перетворюємо класи на числові значення
    [~, class_idx] = ismember(cellstr(predicted), class_names);
    class_grid = reshape(class_idx, size(x1));

    % Створюємо окремий графік
    figure;

    contourf(x1, x2, class_grid, ...
        [0.5 1.5 2.5 3.5], ...
        'LineColor', 'none');

    hold on;

    % Кольори областей класифікації
    colormap([1.0 0.75 0.75;
        0.75 1.0 0.75;
        0.75 0.80 1.0]);

    caxis([0.5 3.5]);

    % Відображаємо квітки
    h = gobjects(3, 1);

    for i = 1:3
        idx = Y_train == class_names{i};

        h(i) = scatter(X_2D(idx,1), X_2D(idx,2), ...
            40, colors(i,:), 'filled', ...
            'MarkerEdgeColor', 'k');
    end

    % Оформлення графіка
    title(sprintf('Межі класифікації KNN при K = %d', k));
    xlabel('Нормалізована довжина чашолистка');
    ylabel('Нормалізована довжина пелюстки');

    legend(h, class_names, 'Location', 'northeast');

    xlim([-0.05 1.05]);
    ylim([-0.05 1.05]);

    grid on;

    % Показуємо пошук сусідів тільки для K = 6
    if k == 6

        % Беремо одну квітку з тестової вибірки
        new_point = X_test_norm(1, [1 3]);

        % Знаходимо 6 найближчих сусідів
        [idx_neighbors, distances] = knnsearch( ...
            X_2D, new_point, 'K', 6);

        % Позначаємо найближчих сусідів
        scatter(X_2D(idx_neighbors,1), ...
            X_2D(idx_neighbors,2), ...
            110, 'k', 'o', 'LineWidth', 2);

        % Радіус кола - відстань до 6-го сусіда
        radius = distances(end);

        % Малюємо пунктирне коло
        theta = linspace(0, 2*pi, 300);

        plot(new_point(1) + radius*cos(theta), ...
            new_point(2) + radius*sin(theta), ...
            'k--', 'LineWidth', 2);

        % Позначаємо нову точку жовтою зірочкою
        scatter(new_point(1), new_point(2), ...
            220, 'p', 'filled', ...
            'MarkerFaceColor', 'y', ...
            'MarkerEdgeColor', 'k');

        % Підписуємо точку
        text(new_point(1)+0.02, new_point(2)+0.03, ...
            'K = 6', 'FontWeight', 'bold');

        % Прогнозуємо вид нової квітки
        predicted_class = predict(model_2D, new_point);

        fprintf('\nПрогноз для нової точки (K = 6): %s\n', ...
            char(predicted_class));

    end

    hold off;

end

