var db = firebase.database();

function writeData(sentence) {
  db.ref("sentences").push(sentence, (err) => console.log(err));
}

export default writeData;
