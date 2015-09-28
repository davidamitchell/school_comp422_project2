brain = require('brain');
fs = require('fs');
_ = require('lodash');

var net = new brain.NeuralNetwork({
   hiddenLayers: [10,10]
});

function reverse(s) {
  return s.split('').reverse().join('');
}

var test  = [];
var train = [];

var filename = 'data/digits00';
var filename = 'data/digits60';

var digits = fs.readFileSync(filename).toString().split("\n");
digits.pop();

var half = (digits.length) / 2
console.log("full size: ", digits.length, " half way: ", half );

train = digits.slice( 0, half )
test = digits.slice( half, digits.length )

// console.log(train.length);
// console.log(train[0]);
// console.log(train[train.length-1]);
//
// console.log(test.length);
// console.log(test[0]);
// console.log(test[test.length-1]);




options = {
  errorThresh: 0.005,  // error threshold to reach
  iterations: 20000,   // maximum training iterations
  log: true,           // console.log() progress periodically
  logPeriod: 500,       // number of iterations between logging
  learningRate: 0.3    // learning rate
}


/* data is in the form of
 *
 * data = [{input: [0, 0], output: [0]}, ...]
 *
 */
var data   = [];
var input  = [];
var output = [];

var line;

for( var i in train ) {

  line  = train[i];
  input = line.split(' ').map(Number);

  var cat = parseInt(input.pop());

  output = {};
  output[cat] = 1;

  // output.push( parseInt(input.pop()) );
  // if (cat == 1) {
    data.push( {input: input, output: output} );
  // }

}

// console.log( data );
// return;
console.log( net.train(data, options) );


var six = "0 1 1 1 1 1 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 1 1 1 1 0".split(' ').map(Number);
var seven = "1 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0".split(' ').map(Number);
var eight = "0 1 1 1 1 1 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 1 1 1 1 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 1 1 1 1 0".split(' ').map(Number);
var nine = "0 1 1 1 1 1 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 1 1 1 1 1 0 9".split(' ').map(Number);

tests = [];
tests.push( six );
tests.push( seven );
tests.push( eight );
tests.push( nine );

for( var i in tests ){
  var x = net.run(tests[i]);
  var m = _.max(x);

  console.log( 5+i, ' -- ', _.invert(x)[m], ' likelyhood: ', m, x );
}


return(0);





for( var i in digits ) {
  var l = digits[i];
  digits[i] = reverse(l);
}


digits = digits.sort();

for( var i in digits ) {
  console.log(digits[i])
}



// var net = new brain.NeuralNetwork();

// data = [{input: [0, 0], output: [0]},
//         {input: [0, 1], output: [1]},
//         {input: [1, 0], output: [1]},
//         {input: [1, 1], output: [0]}];
//
// options = {
//   errorThresh: 0.000005,  // error threshold to reach
//   // iterations: 20000,   // maximum training iterations
//   // log: true,           // console.log() progress periodically
//   // logPeriod: 10,       // number of iterations between logging
//   learningRate: 0.2    // learning rate
// }
//
// console.log( net.train(data, options) );
//
// console.log( "1,0", net.run([1, 0]) );
// console.log( "0,1", net.run([0, 1]) );
// console.log( "0,0", net.run([0, 0]) );
// console.log( "1,1", net.run([0, 0]) );
