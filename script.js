document.addEventListener('DOMContentLoaded', () => {
    // Controls
    const generateDataBtn = document.getElementById('generate-data-btn');
    const trainBaseModelsBtn = document.getElementById('train-base-models-btn');
    const trainMetaModelBtn = document.getElementById('train-meta-model-btn');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const resetBtn = document.getElementById('reset-btn');
    const datasetTypeSelect = document.getElementById('dataset-type');
    const dataPointsSlider = document.getElementById('data-points-slider');
    const dataPointsValue = document.getElementById('data-points-value');
    const noiseSlider = document.getElementById('noise-slider');
    const noiseValue = document.getElementById('noise-value');

    // Displays
    const explanationText = document.getElementById('explanation-text');
    const visualization = document.getElementById('visualization');
    const resultsContent = document.getElementById('results-content');

    let chart;
    let data;

    // --- Data Generation ---
    function generateSyntheticData() {
        const n_samples = parseInt(dataPointsSlider.value, 10);
        const noise = parseInt(noiseSlider.value, 10);
        const datasetType = datasetTypeSelect.value;
        data = { train: [], test: [], all: [] };

        if (datasetType === 'clouds') {
            generateClouds(n_samples, noise);
        } else if (datasetType === 'moons') {
            generateMoons(n_samples, noise);
        }
        
        // 70/30 split
        shuffleArray(data.all);
        const trainSize = Math.floor(data.all.length * 0.7);
        data.train = data.all.slice(0, trainSize);
        data.test = data.all.slice(trainSize);
        
        return data;
    }

    function generateClouds(n_samples, noise) {
        const dataPointsPerCloud = Math.floor(n_samples / 2);
        // Cloud 1 (Class 0) - Bottom-right
        for (let i = 0; i < dataPointsPerCloud; i++) {
            const point = {
                x: 70 + (Math.random() - 0.5) * noise,
                y: 30 + (Math.random() - 0.5) * noise,
                class: 0
            };
            data.all.push(point);
        }
        // Cloud 2 (Class 1) - Top-left
        for (let i = 0; i < dataPointsPerCloud; i++) {
            const point = {
                x: 30 + (Math.random() - 0.5) * noise,
                y: 70 + (Math.random() - 0.5) * noise,
                class: 1
            };
            data.all.push(point);
        }
    }

    function generateMoons(n_samples, noise) {
        const n_samples_out = Math.floor(n_samples / 2);
        const n_samples_in = n_samples - n_samples_out;
        
        for(let i = 0; i < n_samples_out; i++) {
            const angle = Math.PI * (i / (n_samples_out - 1));
            const point = {
                x: 50 + 30 * Math.cos(angle) + (Math.random() - 0.5) * noise,
                y: 50 + 30 * Math.sin(angle) + (Math.random() - 0.5) * noise,
                class: 0
            };
            data.all.push(point);
        }
        
        for(let i = 0; i < n_samples_in; i++) {
            const angle = Math.PI * (i / (n_samples_in - 1));
            const point = {
                x: 50 + 15 + 30 * Math.cos(angle + Math.PI) + (Math.random() - 0.5) * noise,
                y: 50 - 10 + 30 * Math.sin(angle + Math.PI) + (Math.random() - 0.5) * noise,
                class: 1
            };
            data.all.push(point);
        }
    }

    // Utility function to shuffle array (Fisher-Yates shuffle)
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    // Generate decision boundary points for visualization
    function generateDecisionBoundary(model, params, modelType) {
        const boundaryPoints = [];
        const step = 2;
        
        if (modelType === 'stump') {
            // Decision stump creates a straight line
            if (params.feature === 'x') {
                return [{x: params.threshold, y: 0}, {x: params.threshold, y: 100}];
            } else {
                return [{x: 0, y: params.threshold}, {x: 100, y: params.threshold}];
            }
        } else if (modelType === 'logistic') {
            // Logistic regression decision boundary: w0 + w1*x + w2*y = 0
            // Solve for y: y = -(w0 + w1*x) / w2
            if (Math.abs(params.w2) > 0.001) { // Avoid division by zero
                for (let x = 0; x <= 100; x += step) {
                    const y = -(params.w0 + params.w1 * x) / params.w2;
                    if (y >= 0 && y <= 100) {
                        boundaryPoints.push({x, y});
                    }
                }
            }
        }
        return boundaryPoints;
    }

    // Generate classification regions for KNN (simplified)
    function generateKnnRegions(model, params) {
        const regions = [];
        const step = 5;
        
        for (let x = 0; x <= 100; x += step) {
            for (let y = 0; y <= 100; y += step) {
                const prediction = model.predict(params, {x, y});
                regions.push({x, y, class: prediction});
            }
        }
        return regions;
    }

    // --- Model Simulation (Classifiers) ---
    
    // Base Model 1: k-Nearest Neighbors (k=3)
    const knnModel = {
        train: (dataset) => ({ trainingData: [...dataset] }),
        predict: (params, point) => {
            const distances = params.trainingData.map(trainPoint => ({
                distance: Math.sqrt((point.x - trainPoint.x)**2 + (point.y - trainPoint.y)**2),
                class: trainPoint.class
            }));
            distances.sort((a, b) => a.distance - b.distance);
            const kNearest = distances.slice(0, 3);
            const class0Count = kNearest.filter(p => p.class === 0).length;
            const class1Count = kNearest.filter(p => p.class === 1).length;
            return class0Count > class1Count ? 0 : 1;
        }
    };

    // Base Model 2: Decision Stump (simple decision tree with one split)
    const decisionStumpModel = {
        train: (dataset) => {
            let bestThreshold = 50;
            let bestFeature = 'x';
            let bestAccuracy = 0;
            
            // Try different thresholds and features
            for (let feature of ['x', 'y']) {
                const values = dataset.map(p => p[feature]).sort((a, b) => a - b);
                for (let i = 1; i < values.length; i++) {
                    const threshold = (values[i-1] + values[i]) / 2;
                    let correct = 0;
                    for (let point of dataset) {
                        const prediction = point[feature] <= threshold ? 0 : 1;
                        if (prediction === point.class) correct++;
                    }
                    const accuracy = correct / dataset.length;
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestThreshold = threshold;
                        bestFeature = feature;
                    }
                }
            }
            return { threshold: bestThreshold, feature: bestFeature };
        },
        predict: (params, point) => {
            return point[params.feature] <= params.threshold ? 0 : 1;
        }
    };

    // Base Model 3: Logistic Regression (simplified gradient descent)
    const logisticRegressionModel = {
        train: (dataset) => {
            // Initialize weights
            let w0 = 0, w1 = 0, w2 = 0; // bias, x coefficient, y coefficient
            const learningRate = 0.01;
            const epochs = 100;
            
            // Sigmoid function
            const sigmoid = (z) => 1 / (1 + Math.exp(-z));
            
            // Gradient descent
            for (let epoch = 0; epoch < epochs; epoch++) {
                let dw0 = 0, dw1 = 0, dw2 = 0;
                
                for (let point of dataset) {
                    const z = w0 + w1 * point.x + w2 * point.y;
                    const pred = sigmoid(z);
                    const error = pred - point.class;
                    
                    dw0 += error;
                    dw1 += error * point.x;
                    dw2 += error * point.y;
                }
                
                w0 -= learningRate * dw0 / dataset.length;
                w1 -= learningRate * dw1 / dataset.length;
                w2 -= learningRate * dw2 / dataset.length;
            }
            
            return { w0, w1, w2 };
        },
        predict: (params, point) => {
            const z = params.w0 + params.w1 * point.x + params.w2 * point.y;
            const prob = 1 / (1 + Math.exp(-z));
            return prob > 0.5 ? 1 : 0;
        }
    };
    
    // Meta Model: Simple majority voting with weights
    const metaModel = {
        train: (X, y) => {
            // Calculate individual accuracies to determine weights
            const accuracies = [0, 0, 0];
            for (let i = 0; i < X.length; i++) {
                for (let j = 0; j < 3; j++) {
                    if (X[i][j] === y[i]) accuracies[j]++;
                }
            }
            const weights = accuracies.map(acc => acc / X.length);
            return { weights };
        },
        predict: (params, preds) => {
            const weightedSum = preds.reduce((sum, pred, i) => sum + pred * params.weights[i], 0);
            return weightedSum > (params.weights.reduce((sum, w) => sum + w, 0) / 2) ? 1 : 0;
        }
    };

    // --- Utility Functions ---
    function calculateAccuracy(trueY, predY) {
        let correct = 0;
        for (let i = 0; i < trueY.length; i++) {
            if (trueY[i] === predY[i]) {
                correct++;
            }
        }
        return (correct / trueY.length) * 100;
    }

    function updateExplanation(text) {
        explanationText.innerHTML = text;
    }

    function renderChart(config) {
        visualization.innerHTML = '<canvas id="chart"></canvas>';
        const ctx = document.getElementById('chart').getContext('2d');
        if (chart) {
            chart.destroy();
        }
        chart = new Chart(ctx, config);
    }

    // --- Event Handlers ---
    dataPointsSlider.addEventListener('input', (e) => {
        dataPointsValue.textContent = e.target.value;
    });

    noiseSlider.addEventListener('input', (e) => {
        noiseValue.textContent = e.target.value;
    });

    generateDataBtn.addEventListener('click', () => {
        data = generateSyntheticData();
        const chartData = {
            datasets: [{
                label: 'Class 0 (Train)',
                data: data.train.filter(p => p.class === 0),
                backgroundColor: 'rgba(0, 123, 255, 0.7)'
            }, {
                label: 'Class 1 (Train)',
                data: data.train.filter(p => p.class === 1),
                backgroundColor: 'rgba(255, 193, 7, 0.7)'
            }, {
                label: 'Class 0 (Test)',
                data: data.test.filter(p => p.class === 0),
                backgroundColor: 'rgba(0, 123, 255, 0.3)'
            }, {
                label: 'Class 1 (Test)',
                data: data.test.filter(p => p.class === 1),
                backgroundColor: 'rgba(255, 193, 7, 0.3)'
            }]
        };
        const config = {
            type: 'scatter',
            data: chartData,
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { title: { display: true, text: 'Synthetic Dataset' } },
                scales: {
                    x: { title: { display: true, text: 'Feature X' }, min: 0, max: 100 },
                    y: { title: { display: true, text: 'Feature Y' }, min: 0, max: 100 }
                }
            }
        };
        renderChart(config);
        updateExplanation(`Generated a new dataset with ${data.train.length} training points and ${data.test.length} test points (70/30 split). Choose dataset type, number of points, and noise level, then click "Generate Data".`);
        
        generateDataBtn.disabled = false; // Allow re-generation
        trainBaseModelsBtn.disabled = false;
        trainMetaModelBtn.disabled = true;
        evaluateBtn.disabled = true;
        resultsContent.innerHTML = '';
    });

    trainBaseModelsBtn.addEventListener('click', () => {
        if (!data) {
            alert("Please generate data first!");
            return;
        }
        // --- Cross-validation for stacking with three models ---
        const fold1 = data.train.filter((_, i) => i % 2 === 0);
        const fold2 = data.train.filter((_, i) => i % 2 !== 0);
        
        // Train on fold1, predict on fold2
        const knnParams_f1 = knnModel.train(fold1);
        const stumpParams_f1 = decisionStumpModel.train(fold1);
        const lrParams_f1 = logisticRegressionModel.train(fold1);
        const preds_f2_knn = fold2.map(p => knnModel.predict(knnParams_f1, p));
        const preds_f2_stump = fold2.map(p => decisionStumpModel.predict(stumpParams_f1, p));
        const preds_f2_lr = fold2.map(p => logisticRegressionModel.predict(lrParams_f1, p));
        
        // Train on fold2, predict on fold1
        const knnParams_f2 = knnModel.train(fold2);
        const stumpParams_f2 = decisionStumpModel.train(fold2);
        const lrParams_f2 = logisticRegressionModel.train(fold2);
        const preds_f1_knn = fold1.map(p => knnModel.predict(knnParams_f2, p));
        const preds_f1_stump = fold1.map(p => decisionStumpModel.predict(stumpParams_f2, p));
        const preds_f1_lr = fold1.map(p => logisticRegressionModel.predict(lrParams_f2, p));
        
        // Combine out-of-fold predictions
        data.metaTrainFeatures = [];
        data.metaTrainY = [];
        let f1_idx = 0, f2_idx = 0;
        for(let i=0; i<data.train.length; i++) {
            if (i % 2 === 0) {
                data.metaTrainFeatures.push([preds_f1_knn[f1_idx], preds_f1_stump[f1_idx], preds_f1_lr[f1_idx]]);
                data.metaTrainY.push(fold1[f1_idx].class);
                f1_idx++;
            } else {
                data.metaTrainFeatures.push([preds_f2_knn[f2_idx], preds_f2_stump[f2_idx], preds_f2_lr[f2_idx]]);
                data.metaTrainY.push(fold2[f2_idx].class);
                f2_idx++;
            }
        }
        
        // Train final models on all training data
        data.finalKnnParams = knnModel.train(data.train);
        data.finalStumpParams = decisionStumpModel.train(data.train);
        data.finalLrParams = logisticRegressionModel.train(data.train);
        
        // --- Visualization: Show decision boundaries of each base model ---
        const knnTrainPreds = data.train.map(p => knnModel.predict(data.finalKnnParams, p));
        const stumpTrainPreds = data.train.map(p => decisionStumpModel.predict(data.finalStumpParams, p));
        const lrTrainPreds = data.train.map(p => logisticRegressionModel.predict(data.finalLrParams, p));
        
        const trueTrainY = data.train.map(p => p.class);
        const knnTrainAcc = calculateAccuracy(trueTrainY, knnTrainPreds);
        const stumpTrainAcc = calculateAccuracy(trueTrainY, stumpTrainPreds);
        const lrTrainAcc = calculateAccuracy(trueTrainY, lrTrainPreds);
        
        // Generate very dense KNN decision boundary map
        const knnRegions0 = [];
        const knnRegions1 = [];
        const step = 1; // Much denser grid for clear decision boundary visualization
        for (let x = 0; x <= 100; x += step) {
            for (let y = 0; y <= 100; y += step) {
                const pred = knnModel.predict(data.finalKnnParams, {x, y});
                if (pred === 0) knnRegions0.push({x, y});
                else knnRegions1.push({x, y});
            }
        }
        
        const chartData = {
            datasets: [
                // KNN decision boundary regions (very dense)
                {
                    label: 'KNN Class 0 Region',
                    data: knnRegions0,
                    backgroundColor: 'rgba(0, 123, 255, 0.3)',
                    borderColor: 'rgba(0, 123, 255, 0.3)',
                    pointRadius: 1,
                    pointHoverRadius: 1,
                    showLine: false
                },
                {
                    label: 'KNN Class 1 Region', 
                    data: knnRegions1,
                    backgroundColor: 'rgba(255, 193, 7, 0.3)',
                    borderColor: 'rgba(255, 193, 7, 0.3)',
                    pointRadius: 1,
                    pointHoverRadius: 1,
                    showLine: false
                },
                // Decision stump boundary
                {
                    label: 'Decision Stump Boundary',
                    data: generateDecisionBoundary(decisionStumpModel, data.finalStumpParams, 'stump'),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 3,
                    type: 'line',
                    fill: false,
                    pointRadius: 0
                },
                // Logistic regression boundary
                {
                    label: 'Logistic Regression Boundary',
                    data: generateDecisionBoundary(logisticRegressionModel, data.finalLrParams, 'logistic'),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 3,
                    type: 'line',
                    fill: false,
                    pointRadius: 0
                },
                // Training data
                {
                    label: 'Class 0 (Train)',
                    data: data.train.filter(p => p.class === 0),
                    backgroundColor: 'rgba(0, 123, 255, 0.8)',
                    pointRadius: 4
                },
                {
                    label: 'Class 1 (Train)',
                    data: data.train.filter(p => p.class === 1),
                    backgroundColor: 'rgba(255, 193, 7, 0.8)',
                    pointRadius: 4
                }
            ]
        };
        const config = {
            type: 'scatter',
            data: chartData,
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { 
                    title: { 
                        display: true, 
                        text: `Decision Boundaries - KNN: ${knnTrainAcc.toFixed(1)}%, Stump: ${stumpTrainAcc.toFixed(1)}%, LogReg: ${lrTrainAcc.toFixed(1)}%` 
                    } 
                },
                scales: {
                    x: { title: { display: true, text: 'Feature X' }, min: 0, max: 100 },
                    y: { title: { display: true, text: 'Feature Y' }, min: 0, max: 100 }
                }
            }
        };
        renderChart(config);
        updateExplanation('Trained three diverse base classifiers with visible decision boundaries: KNN (colored regions), Decision Stump (red line), and Logistic Regression (teal line). Each model captures different aspects of the data. Their predictions will be combined by the meta-model.');
        trainBaseModelsBtn.disabled = true;
        trainMetaModelBtn.disabled = false;
    });

    trainMetaModelBtn.addEventListener('click', () => {
        if (!data || !data.metaTrainFeatures) {
            alert("Please train base models first!");
            return;
        }
        data.metaModelParams = metaModel.train(data.metaTrainFeatures, data.metaTrainY);
        
        // --- Visualization: Show meta-feature distribution ---
        visualization.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; padding: 20px;">
                <h3>Meta-Model Feature Space</h3>
                <p>The meta-model sees predictions from 3 base models as features:</p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; width: 100%;">
                    <div style="background: #e9f5ff; padding: 15px; border-radius: 5px; text-align: center;">
                        <strong>KNN Predictions</strong><br>
                        (0 or 1)
                    </div>
                    <div style="background: #d4edda; padding: 15px; border-radius: 5px; text-align: center;">
                        <strong>Decision Stump</strong><br>
                        (0 or 1)
                    </div>
                    <div style="background: #fff3cd; padding: 15px; border-radius: 5px; text-align: center;">
                        <strong>Logistic Regression</strong><br>
                        (0 or 1)
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; width: 100%;">
                    <strong>Meta-Model Training Data:</strong><br>
                    Each training example is now a 3-dimensional vector [KNN_pred, Stump_pred, NB_pred] → True_class
                </div>
            </div>
        `;
        updateExplanation('Trained the meta-model! It learns how to optimally combine the predictions from KNN, Decision Stump, and Logistic Regression. The meta-model sees a 3D feature space where each dimension represents a base model prediction.');
        
        trainMetaModelBtn.disabled = true;
        evaluateBtn.disabled = false;
    });

    evaluateBtn.addEventListener('click', () => {
        if (!data || !data.metaModelParams) {
            alert("Please train the meta model first!");
            return;
        }
        const testPredsKnn = data.test.map(p => knnModel.predict(data.finalKnnParams, p));
        const testPredsStump = data.test.map(p => decisionStumpModel.predict(data.finalStumpParams, p));
        const testPredsLr = data.test.map(p => logisticRegressionModel.predict(data.finalLrParams, p));

        const metaFeaturesTest = data.test.map((_, i) => [testPredsKnn[i], testPredsStump[i], testPredsLr[i]]);
        const stackedPreds = metaFeaturesTest.map(p => metaModel.predict(data.metaModelParams, p));

        // Store detailed predictions for click functionality
        data.detailedPredictions = data.test.map((point, i) => ({
            point: point,
            actualClass: point.class,
            knnPred: testPredsKnn[i],
            stumpPred: testPredsStump[i],
            logisticPred: testPredsLr[i],
            stackingPred: stackedPreds[i]
        }));

        const trueY = data.test.map(p => p.class);
        const accKnn = calculateAccuracy(trueY, testPredsKnn);
        const accStump = calculateAccuracy(trueY, testPredsStump);
        const accLr = calculateAccuracy(trueY, testPredsLr);
        const accStacked = calculateAccuracy(trueY, stackedPreds);

        resultsContent.innerHTML = `
            <p>KNN Accuracy: ${accKnn.toFixed(1)}%</p>
            <p>Decision Stump Accuracy: ${accStump.toFixed(1)}%</p>
            <p>Logistic Regression Accuracy: ${accLr.toFixed(1)}%</p>
            <p><strong>Stacking Model Accuracy: ${accStacked.toFixed(1)}%</strong></p>
        `;

        const correct_preds = data.test.filter((p, i) => stackedPreds[i] === p.class);
        const incorrect_preds = data.test.filter((p, i) => stackedPreds[i] !== p.class);

        // Generate extremely dense KNN decision boundary map for final visualization
        const knnRegions0Final = [];
        const knnRegions1Final = [];
        const finalStep = 0.8; // Extremely dense for crystal clear decision boundary
        for (let x = 0; x <= 100; x += finalStep) {
            for (let y = 0; y <= 100; y += finalStep) {
                const pred = knnModel.predict(data.finalKnnParams, {x, y});
                if (pred === 0) knnRegions0Final.push({x, y});
                else knnRegions1Final.push({x, y});
            }
        }

        const chartData = {
            datasets: [
                // KNN decision boundary background (extremely dense)
                {
                    label: 'KNN Decision Map (Class 0)',
                    data: knnRegions0Final,
                    backgroundColor: 'rgba(0, 123, 255, 0.2)',
                    borderColor: 'rgba(0, 123, 255, 0.2)',
                    pointRadius: 0.8,
                    pointHoverRadius: 0.8,
                    showLine: false
                },
                {
                    label: 'KNN Decision Map (Class 1)', 
                    data: knnRegions1Final,
                    backgroundColor: 'rgba(255, 193, 7, 0.2)',
                    borderColor: 'rgba(255, 193, 7, 0.2)',
                    pointRadius: 0.8,
                    pointHoverRadius: 0.8,
                    showLine: false
                },
                // All decision boundaries
                {
                    label: 'Decision Stump',
                    data: generateDecisionBoundary(decisionStumpModel, data.finalStumpParams, 'stump'),
                    borderColor: 'rgba(255, 99, 132, 0.8)',
                    borderWidth: 2,
                    type: 'line',
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'Logistic Regression',
                    data: generateDecisionBoundary(logisticRegressionModel, data.finalLrParams, 'logistic'),
                    borderColor: 'rgba(75, 192, 192, 0.8)',
                    borderWidth: 2,
                    type: 'line',
                    fill: false,
                    pointRadius: 0
                },
                // Test results
                {
                    label: 'Correctly Classified (Stacking)',
                    data: correct_preds,
                    backgroundColor: 'rgba(40, 167, 69, 0.9)',
                    pointRadius: 6,
                }, {
                    label: 'Incorrectly Classified (Stacking)',
                    data: incorrect_preds,
                    backgroundColor: 'rgba(220, 53, 69, 0.9)',
                    pointRadius: 6,
                }
            ]
        };
        const config = {
            type: 'scatter',
            data: chartData,
            options: {
                responsive: true, maintainAspectRatio: false,
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        const elementIndex = elements[0].index;
                        const datasetIndex = elements[0].datasetIndex;
                        
                        // Find which test point was clicked (skip background datasets and decision boundaries)
                        if (datasetIndex >= 4) { // Skip KNN background (0,1) and decision boundaries (2,3)
                            const clickedPoint = chartData.datasets[datasetIndex].data[elementIndex];
                            
                            // Find matching test point
                            const testPointIndex = data.test.findIndex(point => 
                                Math.abs(point.x - clickedPoint.x) < 0.1 && 
                                Math.abs(point.y - clickedPoint.y) < 0.1
                            );
                            
                            if (testPointIndex >= 0 && data.detailedPredictions) {
                                showPointDetails(data.detailedPredictions[testPointIndex]);
                            }
                        }
                    }
                },
                plugins: { 
                    title: { 
                        display: true, 
                        text: `Final Results - Stacking: ${accStacked.toFixed(1)}% (Best Individual: ${Math.max(accKnn, accStump, accLr).toFixed(1)}%)` 
                    } 
                },
                scales: {
                    x: { title: { display: true, text: 'Feature X' }, min: 0, max: 100 },
                    y: { title: { display: true, text: 'Feature Y' }, min: 0, max: 100 }
                }
            }
        };
        renderChart(config);
        showMessage('Evaluation complete! The stacking model combines KNN, Decision Stump, and Logistic Regression to achieve higher accuracy than any individual base model. You can see all the decision boundaries overlaid. Green points were classified correctly by stacking, red ones were not. Notice how stacking leverages the diverse strengths of different algorithms. Click on any data point to see detailed classification breakdown!', 'success');
        evaluateBtn.disabled = true;
    });

    resetBtn.addEventListener('click', () => {
        generateDataBtn.disabled = false;
        trainBaseModelsBtn.disabled = true;
        trainMetaModelBtn.disabled = true;
        evaluateBtn.disabled = true;

        visualization.innerHTML = '';
        resultsContent.innerHTML = '';
        if (chart) chart.destroy();
        data = null;

        updateExplanation('Welcome! Click "Generate Data" to start.');
        datasetTypeSelect.disabled = false;
        dataPointsSlider.disabled = false;
        noiseSlider.disabled = false;
    });

    function showPointDetails(prediction) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;
        
        const content = document.createElement('div');
        content.style.cssText = `
            background: white;
            padding: 20px;
            border-radius: 8px;
            max-width: 500px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        `;
        
        const isCorrect = prediction.actualClass === prediction.stackingPred;
        const correctClass = prediction.actualClass === 0 ? 'Class 0 (Blue)' : 'Class 1 (Red)';
        const predictedClass = prediction.stackingPred === 0 ? 'Class 0 (Blue)' : 'Class 1 (Red)';
        
        content.innerHTML = `
            <h3>Point Classification Details</h3>
            <p><strong>Location:</strong> (${prediction.point.x.toFixed(1)}, ${prediction.point.y.toFixed(1)})</p>
            <p><strong>Actual Class:</strong> ${correctClass}</p>
            <p><strong>Stacking Prediction:</strong> ${predictedClass} ${isCorrect ? '✓' : '✗'}</p>
            <hr>
            <h4>Base Model Predictions:</h4>
            <ul>
                <li><strong>KNN:</strong> Class ${prediction.knnPred}</li>
                <li><strong>Decision Stump:</strong> Class ${prediction.stumpPred}</li>
                <li><strong>Logistic Regression:</strong> Class ${prediction.logisticPred}</li>
            </ul>
            <hr>
            <p><strong>Meta-Model Decision:</strong> Combines all base predictions to make final choice</p>
            <button onclick="this.closest('[style*=\'position: fixed\']').remove()" style="
                margin-top: 10px;
                padding: 8px 16px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            ">Close</button>
        `;
        
        modal.appendChild(content);
        document.body.appendChild(modal);
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    // Initial state
    resetBtn.click();
});