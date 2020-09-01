var app = require('readline-sync');

var firstNumber = app.questionInt('please enter one number');
var secondNumber = app.questionInt('please ente another number');
var operators = app.question('please enter an operator(+,-,/,*)');
var result = 0;
if(operators === "+"){
  result = firstNumber+ secondNumber;
  console.log(result);
} else if (operators === "-") {
  result = firstNumber - secondNumber;
  console.log(result);
} else if (operators === "/") {
  result = firstNumber / secondNumber;
  console.log(result);
} else if (operators === "*") {
  result = firstNumber * secondNumber;
  console.log(result);
} else {
  console.log("please choose an operator");
}
