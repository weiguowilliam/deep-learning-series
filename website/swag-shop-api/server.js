var express = require('express');
var app = express();
var bodyParser = require('body-parser');
var mongoose = require('mongoose');
var db = mongoose.connect('mongodb://localhost/swag-shop');

var Product = require('./model/product');
var Wishlist = require('./model/wishlist');

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: false}));


app.post('/product', function(request, response) {
    var product = new Product();
    product.title = request.body.title;
    product.price = request.body.price;
    product.save(function(err, savedProduct) {
       if (err) {
           response.status(500).send({error:"Could not save product"});
       } else {
           response.send(savedProduct);
       }
    });
});

app.get('/product',function(req,response){
  Product.find({},function(err,products){
    if (err) {
      response.status(500).send({error:"Coule not fecth products"});
    } else {
      response.send(products);
    }
  });
});

app.get('/wishlist',function(req,res){
  Wishlist.find({},function(err,wishlists){
    res.send(wishlists);
});
});

app.post('/wishlist',function(req,res){
  var wishlist = new Wishlist();
  wishlist.title = req.body.title;

  wishlist.save(function(err,saveWishlist){
    if (err) {
      res.status(500).send({error:"couldnt save wishlist"});
    } else{
      res.send(saveWishlist);
    }
  });
});

app.put('/wishlist/product/add',function(request,response){
  Product.findOne({_id:request.body.productId},function(err,product){
    if (err) {
      response.status(500).send({error:"couldnt add product to wishlist 1"});
    } else {
    //  Wishlist.update({_id:request.body.wishListId}, {$addToSet:{products: product._id}}, function(err, wishlist) {
      Wishlist.update({_id:request.body.wishListId}, {$addToSet:{products: product._id}}, function(err, wishlist) {
        if (err) {
            response.status(500).send({error:"couldnt add product to wishlist 2"});
        } else {
          response.send(wishlist);
        }
      });
    }
  });
});







app.listen(3000,function(){
  console.log("swag shop api running on port 3000..");
});
