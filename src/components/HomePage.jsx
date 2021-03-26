import React, { Component } from "react";
import img1 from "./images/img1.jpg";
import img2 from "./images/img2.jpg";
import SuggestionCard from "./SuggestionCard";
import Keyboard from "./Keyboard";
import { FaKeyboard } from "react-icons/fa";
import { RiMenuUnfoldLine } from "react-icons/ri";
import dictionary from "./virtualKeyboardDictionary";
import SideNav from "./SideNav";
import diffCalculator from "./diffCalculator";
class HomePage extends Component {
  state = {
    imgHolder: img1,
    keyboard: true,
    corrections: [["Exmple", "Example"]],
  };

  constructor(props) {
    super(props);
    this.onChangeContent = this.onChangeContent.bind(this);
    this.onEditTitle = this.onEditTitle.bind(this);
    this.toggleKeyboard = this.toggleKeyboard.bind(this);
    this.calculateDifference = this.calculateDifference.bind(this);
    this.deleteSuggestionCard = this.deleteSuggestionCard.bind(this);
    this.checkSentence = this.checkSentence.bind(this);
  }
  componentDidMount() {
    // this.calculateDifference(
    //   "Hi I am Shreyas Nikam. I am a Front End Engineer.",
    //   "Hello."
    // );
  }
  replaceContent(input, target, count) {
    let result = document.getElementById("content").value;
    document.getElementById("content").value = result.replace(input, target);
    console.log(input, target);
    //     let temp = result.slice(0, count);
    // console.log("before: ", temp);
    // temp += target;
    // console.log("replace: ", temp);

    // temp += result.slice(count + input.length, result.length);
    // console.log("after: ", temp);
    // document.getElementById("content").value = temp;
  }
  calculateDifference(input, target) {
    let corrections = [];
    let difference = diffCalculator(input, target);
    let count = 0;
    for (let i = 0; i < difference.length - 1; i++) {
      if (difference[i][0] === 0) {
        count += difference[i][1].length;
      } else if (difference[i][0] === -1) {
        if (difference[i + 1][0] === 1)
          corrections.push([difference[i][1], difference[i + 1][1], count]);
        else corrections.push([difference[i][1], "", count]);
        i += 1;
        count += difference[i][1].length;
      } else {
        console.log(difference[i][1]);
        corrections.push(["", difference[i][1], count]);
        count += 1;
      }
    }
    this.setState({ corrections: corrections });
    console.log(corrections, this.state.corrections);
  }
  onEditTitle(e) {
    document.getElementsByTagName("title")[0].innerHTML = e.target.value;
  }
  toggleKeyboard() {
    if (this.state.keyboard) {
      document.getElementById("keyboard_holder").style.display = "none";
      document.getElementById("content").rows = 20;
    } else {
      document.getElementById("keyboard_holder").style.display = "block";
      document.getElementById("content").rows = 8;
    }
    this.setState({ keyboard: !this.state.keyboard });
  }
  checkSentence() {
    this.setState({ isLoading: true });
    let sentence = document.getElementById("content").value;
    // console.log(typeof sentence);
    fetch("http://localhost:9696/suggest", {
      method: "POST",
      body: JSON.stringify({
        input_sentence: sentence,
      }),
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    })
      .then((res) => res.json())
      .then((res) => {
        console.log(res);
        this.setState({ corrections: [[sentence, res["output"]]] });
        console.log(this.state);
        this.setState({ isLoading: false });
      });
  }
  onChangeContent(e) {
    e.preventDefault();
    let temp = e.target.value;
    let ct = e.which || e.keyCode;
    temp += dictionary[ct];
    e.target.value = temp;
    if (temp === "") {
      document.getElementById("words").innerHTML = "0 words";
      document.getElementById("characters").innerHTML = "0 characters";
    } else {
      let s = temp;
      s = s.replace(/(^s*)|(s*$)/gi, "");
      s = s.replace(/[ ]{2,}/gi, " ");
      s = s.replace(/ +(?= )/g, "");
      s = s.replace(/\n /, "\n");
      document.getElementById("words").innerHTML =
        s.split(" ").filter((e) => {
          return e !== "";
        }).length + " words";

      document.getElementById("characters").innerHTML =
        temp.length + " characters";
    }
  }
  openSideNav() {
    document.getElementById("mySidenav").style.width = "250px";
  }
  deleteSuggestionCard(id) {
    // document
    //   .getElementById("suggestions_content")
    //   .removeChild(document.getElementById(id));

    let temp = parseInt(id.replace("suggestion_", ""));
    console.log(temp);
    let corrections = this.state.corrections;
    corrections.splice(temp, 1);
    this.setState({ corrections: corrections });
  }
  render() {
    return (
      <div className="container-fluid">
        <SideNav />
        <div className="row">
          <div className="col-sm-1">
            <br />
            <div
              id="sidenav_holder"
              style={{ width: "100px", whiteSpace: "nowrap" }}
            >
              <FaKeyboard
                onClick={this.toggleKeyboard}
                fill="rgb(83, 6, 226)"
                size="20px"
                title="Virtual Keyboard"
                style={{ cursor: "pointer" }}
                data-toggle="tooltip"
              />
              <span
                style={{
                  marginLeft: "8px",
                  borderLeft: "1px solid lightgray",
                }}
              ></span>
              <RiMenuUnfoldLine
                fill="grey"
                size="16px"
                title="Open Menu"
                data-toggle="tooltip2"
                onClick={this.openSideNav}
                style={{
                  margin: "0.4vw",
                  cursor: "pointer",
                }}
              />
            </div>
          </div>
          <div className="col-sm-6" id="doc">
            <input
              type="text"
              className="form-control doc_title m-2"
              placeholder="Untitled Document"
              onBlur={(e) => this.onEditTitle(e)}
              id="doc_title"
            />{" "}
            <textarea
              rows="8"
              id="content"
              onKeyPress={(e) => this.onChangeContent(e)}
              className="form-control content m-2 mt-5"
              placeholder="Type or paste (Ctrl+V) your text here."
            />
            <div id="keyboard_holder">
              <Keyboard />
            </div>
            <button
              className="btn btn-sm btn-primary m-1"
              onClick={this.checkSentence}
            >
              Check!
            </button>
            <div className="float-right">
              <small id="characters" className="text-secondary m-1">
                0 characters
              </small>
              <small id="words" className="text-secondary m-1">
                0 words
              </small>
            </div>
          </div>
          <div className="col-sm-4">
            <div className="doc_title m-3 suggestions_title">
              All suggestions
            </div>
            {this.state.isLoading ? <>Loading...</> : <></>}
            <div className="suggestions_content" id="suggestions_content">
              {this.state.corrections.map((e, i) => (
                <SuggestionCard
                  key={i}
                  title={e[0]}
                  correction={e[1]}
                  index={e[2]}
                  id={"suggestion_" + i}
                  deleteSuggestionCard={this.deleteSuggestionCard}
                  replaceContent={this.replaceContent}
                />
              ))}
              {this.state.corrections.length === 0 ? (
                document.getElementById("content").value === "" ? (
                  <>
                    <img src={img1} width="100%" alt="placeholder" />
                    <center>
                      <small id="s1" className="s1">
                        Nothing to check yet
                      </small>
                      <br />
                      <small id="s2" className="s2">
                        Start writing to check for suggestions and feedback.
                      </small>
                    </center>
                  </>
                ) : (
                  <>
                    <img src={img2} width="100%" alt="placeholder" />
                    <center>
                      <small id="s1" className="s1">
                        No issues found{" "}
                      </small>
                      <br />
                      <small id="s2" className="s2">
                        We ran many checks on your content and found no writing
                        issues. <br /> Check back when you're ready to write
                        some more.{" "}
                      </small>
                    </center>
                  </>
                )
              ) : (
                <></>
              )}
            </div>
          </div>

          <iframe
            id="printing-frame"
            name="print_frame"
            src="about:blank"
            title="print_frame"
            style={{ display: "none" }}
          ></iframe>
        </div>
      </div>
    );
  }
}

export default HomePage;
