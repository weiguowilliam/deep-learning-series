//x is what percent of y
var numField1 = document.getElementById('numField1');
var numField2 = document.getElementById('numField2');
var resultField = document.getElementById('resultField');
var form = document.getElementById('xIsWhatPercentOfY');


form.addEventListener('submit',function(event){
    if((!numField1.value)||(!numField1.value)){
        alert("No value");
    } else{
    var x = parseFloat(numField1.value);
    var y = parseFloat(numField2.value);
    var result = 100 * x/y;
        resultField.innerText ="Answer: "+ result+"%";
        event.preventDefault();
    }
});

// what is y percent of x
var x2 = document.getElementById("x2");
var y2 = document.getElementById("y2");
var result2 = document.getElementById("result2");
var form2 = document.getElementById("yPercentOfX");

form2.addEventListener("submit",function(event){
     if((!x2.value)||(!y2.value)){
        alert("No value");
    } else{
    var x = parseFloat(x2.value);
    var y = parseFloat(y2.value);
    var result = x*y/100;
        result2.innerText ="Answer: "+ result+"%";
        event.preventDefault();
    }
});