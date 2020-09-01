function stu(firstname,lastname,age){
    this.firstname = firstname;
    this.lastname = lastname;
    this.age = age;
    this.greeting = function(){
        return 'Hey '+this.lastname;
    };
}

var s1 = new stu("a","b",3);
var stu1 = [];

stu1.push(s1);
stu1.push(new stu("a","b","4"));

for (var key in s1){
    console.log(s1[key]);
}
