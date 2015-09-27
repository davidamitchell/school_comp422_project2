brain = require('brain');

var net = new brain.NeuralNetwork({
   hiddenLayers: [10]
});

// var net = new brain.NeuralNetwork();

data = [{input: [0, 0], output: [0]},
        {input: [0, 1], output: [1]},
        {input: [1, 0], output: [1]},
        {input: [1, 1], output: [0]}];

options = {
  errorThresh: 0.000005,  // error threshold to reach
  // iterations: 20000,   // maximum training iterations
  log: true,           // console.log() progress periodically
  logPeriod: 10,       // number of iterations between logging
  learningRate: 0.2    // learning rate
}

console.log( net.train(data, options) );

console.log( "1,0", net.run([1, 0]) );
console.log( "0,1", net.run([0, 1]) );
console.log( "0,0", net.run([0, 0]) );
console.log( "1,1", net.run([0, 0]) );
