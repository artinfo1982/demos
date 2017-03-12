var MongoClient = require("mongodb").MongoClient;

var DB = function () {};

DB.prototype.insert = function (url, usrName, usrPwd, colName, data) {
    MongoClient.connect(url, function (err, db) {
        if (err) {
            console.log("|ERROR|insert()|connect to mongodb failed|" + err);
        }
        var adminDb = db.admin();
        adminDb.authenticate(usrName, usrPwd, function (err) {
            if (err) {
                console.log("|ERROR|insert()|mongodb authenticate failed|" + err);
                db.close();
            }
        });
        db.collection(colName, function (err, col) {
            if (err) {
                console.log("|ERROR|insert()|get mongodb collection failed|" + err);
                db.close();
            }
            col.insert(data, function (err) {
                if (err) {
                    console.log("|ERROR|insert()|insert into mongodb failed|" + err);
                    db.close();
                }
            });
            col.findOne(data, function (err, result) {
                if (err) {
                    console.log("|ERROR|insert()|cannot find data in mongodb|" + err);
                    db.close();
                }
                else if (result && db) {
                    db.close();
                    console.log("|INFO|insert()|insert data into mongodb successfully");
                }
            });
        });
    });
};

DB.prototype.delete = function (url, usrName, usrPwd, colName, data) {
    MongoClient.connect(url, function (err, db) {
        if (err) {
            console.log("|ERROR|delete()|connect to mongodb failed|" + err);
        }
        var adminDb = db.admin();
        adminDb.authenticate(usrName, usrPwd, function (err) {
            if (err) {
                console.log("|ERROR|delete()|mongodb authenticate failed|" + err);
                db.close();
            }
        });
        db.collection(colName, function (err, col) {
            if (err) {
                console.log("|ERROR|delete()|get mongodb collection failed|" + err);
                db.close();
            }
            col.findOne(data, function (err, result) {
                if (err) {
                    console.log("|ERROR|delete()|cannot find data which will be deleted in mongodb|" + err);
                    db.close();
                }
                if (result) {
                    col.remove(data, function (err) {
                        if (err) {
                            console.log("|ERROR|delete()|remove data from mongodb failed|" + err);
                            db.close();
                        }
                        else if (db) {
                            db.close();
                            console.log("|INFO|delete()|remove data from mongodb successfully");
                        }
                    });
                }
            });
        });
    });
};

module.exports = new DB();
