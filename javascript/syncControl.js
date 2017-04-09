


var async = require("async");

var a = function () {
    for (var i=0; i<10; i++)
    {
        console.log("a");
    }
};

var b = function () {
    for (var i=0; i<10; i++)
    {
        console.log("b");
    }
};

var c = function () {
    for (var i=0; i<10; i++)
    {
        console.log("c");
    }
};

async.series
(
    [
        function (callback) {
            callback(a());
        },
        function (callback) {
            callback(b());
        },
        function (callback) {
            callback(c());
        }
    ]
);
