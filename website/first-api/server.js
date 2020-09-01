var app = require('express')();
var bodyParser = require('body-parser');

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: false}));

var ingredients = [
  {
    "id":"a",
    "text":"egg"
  },
  {
    "id":"b",
    "text":"milk"
  },
  {
    "id":"c",
    "text":"bacon"
  }
];

app.listen(3000,function(){
  console.log("First api running on port 3000!");
});

app.get('/ingredients',function(request, response){
  response.send(ingredients);
});

app.get('/funions',function(req,res){
  res.send("yo show me!");
});



app.post('/ingredients',function(req,res){
  var ingredient = req.body;
  if (!ingredient || ingredient.text === ""){
    res.status(500).send({error: "Your ingredient must have a name"});
  } else{
    ingredients.push(ingredient);
    res.status(200).send(ingredient);
  }
});

app.put('/ingredients/:ingredientId',function(request,response){
  var newText = request.body.text;
  if(!newText || newText === "" ){
    response.status(500).send({error:"you must provide ingredient text."})
  } else{
    var objectFound = false;
    for(var x= 0; x < ingredients.length;x++){
      var ing = ingredients[x];
      if(ing.id === request.params.ingredientId){
        ingredients[x].text = newText;
        objectFound = true;
        break;
      }
    }
    if(!objectFound){
      response.send({error:"not found"});
    } else{
      response.send(ingredients);
    }
  }


});
