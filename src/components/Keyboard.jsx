import React, { Component } from "react";

class Keyboard extends Component {
  toggle1 = [
    " ॊ",
    "१",
    "२",
    "३",
    "४",
    "५",
    "६",
    "७",
    "८",
    "९",
    "०",
    "-",
    " ृ",
    " ौ",
    " ै",
    " ा",
    " ी",
    " ू",
    "ब",
    "ह",
    "ग",
    "द",
    "ज",
    "ड",
    " ़",
    " ॉ",
    " ो",
    " े",
    " ्",
    " ि",
    " ु",
    "प",
    "र",
    "क",
    "त",
    "च",
    "ट",
    " ॆ",
    " ं",
    "म",
    "न",
    "व",
    "ल",
    "स",
    ",",
    ".",
    "य",
  ];
  toggle2 = [
    "ऒ",
    "ऍ",
    " ॅ",
    " ्र",
    "र्",
    "ज्ञ",
    "त्र",
    "क्ष",
    "श्र",
    "(",
    ")",
    " ः",
    "ऋ",
    "औ",
    "ऐ",
    "आ",
    "ई",
    "ऊ",
    "भ",
    "ङ",
    "घ",
    "ध",
    "झ",
    "ढ",
    "ञ",
    "ऑ",
    "ओ",
    "ए",
    "अ",
    "इ",
    "उ",
    "फ",
    "ऱ",
    "ख",
    "थ",
    "छ",
    "ठ",
    "ऎ",
    " ँ",
    "ण",
    "ऩ",
    "ऴ",
    "ळ",
    "श",
    "ष",
    "।",
    "य़",
  ];
  state = { current: this.toggle1, baseClass: "base", cur: 0 };
  constructor(props) {
    super(props);
    this.componentDidMount = this.componentDidMount.bind(this);
    this.changeModifierState = this.changeModifierState.bind(this);
  }
  changeModifierState() {
    console.log("clicked");
    this.state.cur == 0
      ? this.setState({ current: this.toggle2, baseClass: "raised", cur: 1 })
      : this.setState({ current: this.toggle1, baseClass: "base", cur: 0 });
    console.log(this.state);
  }
  componentDidMount() {
    document.onkeypress = function (evt) {
      evt = evt || window.event;
      if (evt.getModifierState("CapsLock") === true) {
        this.setState({
          current: this.toggle2,
        });
      }
      if (evt.getModifierState("Shift")) {
        console.log("hi");
        this.setState({
          current: this.toggle1,
        });
      }
    }.bind(this);
  }
  buttonFunction(id) {
    document.getElementById("content").value +=
      this.state.cur == 0 ? this.toggle1[id] : this.toggle2[id];
  }
  render() {
    return (
      <div className="panel-body">
        <ul className="keyboard">
          <li onClick={() => this.buttonFunction(0)} className={" char"}>
            {this.toggle2[0]}
            <br />
            <span>{this.toggle1[0]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(1)} className={" char"}>
            {this.toggle2[1]}
            <br />
            <span>{this.toggle1[1]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(2)} className={" char"}>
            {this.toggle2[2]}
            <br />
            <span>{this.toggle1[2]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(3)} className={" char"}>
            {this.toggle2[3]}
            <br />
            <span>{this.toggle1[3]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(4)} className={" char"}>
            {this.toggle2[4]}
            <br />
            <span>{this.toggle1[4]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(5)} className={" char"}>
            {this.toggle2[5]}
            <br />
            <span>{this.toggle1[5]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(6)} className={" char"}>
            {this.toggle2[6]}
            <br />
            <span>{this.toggle1[6]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(7)} className={" char"}>
            {this.toggle2[7]}
            <br />
            <span>{this.toggle1[7]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(8)} className={" char"}>
            {this.toggle2[8]}
            <br />
            <span>{this.toggle1[8]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(9)} className={" char"}>
            {this.toggle2[9]}
            <br />
            <span>{this.toggle1[9]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(10)} className={" char"}>
            {this.toggle2[10]}
            <br />
            <span>{this.toggle1[10]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(11)} className={" char"}>
            {this.toggle2[11]}
            <br />
            <span>{this.toggle1[11]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(12)} className={" char"}>
            {this.toggle2[12]}
            <br />
            <span>{this.toggle1[12]}</span>{" "}
          </li>
          <li
            className="backspace last"
            style={{
              height: "50px",
              verticalAlign: "center",
              paddingTop: "12.5px",
            }}
          >
            Backspace
          </li>
          <li
            className="tab"
            style={{
              height: "50px",
              verticalAlign: "center",
              paddingTop: "12.5px",
            }}
          >
            Tab
          </li>
          <li onClick={() => this.buttonFunction(13)} className={" char"}>
            {this.toggle2[13]}
            <br />
            <span>{this.toggle1[13]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(14)} className={" char"}>
            {this.toggle2[14]}
            <br />
            <span>{this.toggle1[14]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(15)} className={" char"}>
            {this.toggle2[15]}
            <br />
            <span>{this.toggle1[15]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(16)} className={" char"}>
            {this.toggle2[16]}
            <br />
            <span>{this.toggle1[16]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(17)} className={" char"}>
            {this.toggle2[17]}
            <br />
            <span>{this.toggle1[17]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(18)} className={" char"}>
            {this.toggle2[18]}
            <br />
            <span>{this.toggle1[18]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(19)} className={" char"}>
            {this.toggle2[19]}
            <br />
            <span>{this.toggle1[19]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(20)} className={" char"}>
            {this.toggle2[20]}
            <br />
            <span>{this.toggle1[20]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(21)} className={" char"}>
            {this.toggle2[21]}
            <br />
            <span>{this.toggle1[21]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(22)} className={" char"}>
            {this.toggle2[22]}
            <br />
            <span>{this.toggle1[22]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(23)} className={" char"}>
            {this.toggle2[23]}
            <br />
            <span>{this.toggle1[23]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(24)} className={" char"}>
            {this.toggle2[24]}
            <br />
            <span>{this.toggle1[24]}</span>{" "}
          </li>
          <li
            onClick={() => this.buttonFunction(25)}
            className={" char"}
            style={{ width: "9%" }}
          >
            {this.toggle2[25]}
            <br />
            <span>{this.toggle1[25]}</span>{" "}
          </li>
          <li
            className={" capslock"}
            style={{
              height: "50px",
              verticalAlign: "center",
              paddingTop: "12.5px",
            }}
          >
            Caps
          </li>
          <li onClick={() => this.buttonFunction(26)} className={" char"}>
            {this.toggle2[26]}
            <br />
            <span>{this.toggle1[26]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(27)} className={" char"}>
            {this.toggle2[27]}
            <br />
            <span>{this.toggle1[27]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(28)} className={" char"}>
            {this.toggle2[28]}
            <br />
            <span>{this.toggle1[28]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(29)} className={" char"}>
            {this.toggle2[29]}
            <br />
            <span>{this.toggle1[29]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(30)} className={" char"}>
            {this.toggle2[30]}
            <br />
            <span>{this.toggle1[30]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(31)} className={" char"}>
            {this.toggle2[31]}
            <br />
            <span>{this.toggle1[31]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(32)} className={" char"}>
            {this.toggle2[32]}
            <br />
            <span>{this.toggle1[32]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(33)} className={" char"}>
            {this.toggle2[33]}
            <br />
            <span>{this.toggle1[33]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(34)} className={" char"}>
            {this.toggle2[34]}
            <br />
            <span>{this.toggle1[34]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(35)} className={" char"}>
            {this.toggle2[35]}
            <br />
            <span>{this.toggle1[35]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(36)} className={" char"}>
            {this.toggle2[36]}
            <br />
            <span>{this.toggle1[36]}</span>{" "}
          </li>
          <li
            className="return last"
            style={{
              height: "50px",
              verticalAlign: "center",
              paddingTop: "12.5px",
            }}
          >
            Enter
          </li>
          <li
            onClick={this.changeModifierState}
            className={this.state.baseClass + " shift"}
            style={{
              height: "50px",
              verticalAlign: "center",
              paddingTop: "12.5px",
            }}
          >
            Shift
          </li>
          <li onClick={() => this.buttonFunction(37)} className={" char"}>
            {this.toggle2[37]}
            <br />
            <span>{this.toggle1[37]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(38)} className={" char"}>
            {this.toggle2[38]}
            <br />
            <span>{this.toggle1[38]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(39)} className={" char"}>
            {this.toggle2[39]}
            <br />
            <span>{this.toggle1[39]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(40)} className={" char"}>
            {this.toggle2[40]}
            <br />
            <span>{this.toggle1[40]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(41)} className={" char"}>
            {this.toggle2[41]}
            <br />
            <span>{this.toggle1[41]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(42)} className={" char"}>
            {this.toggle2[42]}
            <br />
            <span>{this.toggle1[42]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(43)} className={" char"}>
            {this.toggle2[43]}
            <br />
            <span>{this.toggle1[43]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(44)} className={" char"}>
            {this.toggle2[44]}
            <br />
            <span>{this.toggle1[44]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(45)} className={" char"}>
            {this.toggle2[45]}
            <br />
            <span>{this.toggle1[45]}</span>{" "}
          </li>
          <li onClick={() => this.buttonFunction(46)} className={" char"}>
            {this.toggle2[46]}
            <br />
            <span>{this.toggle1[46]}</span>{" "}
          </li>
          <li
            onClick={this.changeModifierState}
            className={this.state.baseClass + " shift"}
            style={{
              height: "50px",
              verticalAlign: "center",
              paddingTop: "12.5px",
            }}
          >
            Shift
          </li>
          <li
            className="space"
            style={{
              height: "50px",
              verticalAlign: "center",
              paddingTop: "12.5px",
            }}
          >
            Space
          </li>
        </ul>
      </div>
    );
  }
}

export default Keyboard;
