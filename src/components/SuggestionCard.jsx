import React, { Component } from "react";
import { MdClear, MdExpandLess, MdExpandMore, MdFlag } from "react-icons/md";
import { FcCheckmark } from "react-icons/fc";
class SuggestionCard extends Component {
  state = {};
  constructor(props) {
    super(props);
    this.changeContent = this.changeContent.bind(this);
    this.expandDiv = this.expandDiv.bind(this);
    this.minimizeDiv = this.minimizeDiv.bind(this);
  }
  sampleText = [
    "Consider making the following changes:",
    "Our model predicted the following correction for your sentence.",
    "There is an error in your sentence. Consider changing:",
    "Oops! You made a blunder: ",
    "Check out the following suggestion:",
  ];
  changeContent() {
    this.props.replaceContent(
      this.props.title,
      this.props.correction,
      this.props.index
    );
    // document.getElementById("content").value = temp;
    this.props.deleteSuggestionCard(this.props.id);
  }
  minimizeDiv() {
    document.getElementById(this.props.id + "_minimized_div").style.display =
      "block";
    document.getElementById(this.props.id + "_expanded_div").style.display =
      "none";
  }
  expandDiv() {
    document.getElementById(this.props.id + "_minimized_div").style.display =
      "none";
    document.getElementById(this.props.id + "_expanded_div").style.display =
      "block";
  }

  render() {
    return (
      <div className="cardx" id={this.props.id}>
        <div className="cardx-body">
          <div id={this.props.id + "_minimized_div"} onClick={this.expandDiv}>
            <div className="row">
              <div className="col-sm-10">
                <span className="circle"></span>
                <span className="circle2"></span>
                <span style={{ marginLeft: "15px" }} className="error_title">
                  {this.props.title}
                </span>
              </div>
              <div className="col-sm-1 float_right">
                <MdExpandMore
                  size={20}
                  fill="lightgrey"
                  data-toggle="tooltip"
                  title="Expand"
                  className="mr-2"
                  onClick={this.expandDiv}
                />
              </div>
            </div>
          </div>
          <div id={this.props.id + "_expanded_div"} style={{ display: "none" }}>
            <div className="row">
              <div className="col-sm-10">
                <div>
                  <span className="circle"></span>
                  <span className="circle2"></span>
                  <span style={{ marginLeft: "15px" }} className="error_title">
                    {this.props.title}
                  </span>
                  <br />
                  <div style={{ marginLeft: "15px", marginTop: "5px" }}>
                    <small className="text-secondary">
                      {
                        this.sampleText[
                          Math.floor(Math.random() * this.sampleText.length)
                        ]
                      }
                    </small>
                    <br />
                    <span className="error_text" style={{ marginTop: "15px" }}>
                      {this.props.title}
                    </span>{" "}
                    →{" "}
                    <span className="correct_text">
                      {this.props.correction}
                    </span>
                    <br />
                  </div>
                </div>
              </div>
              <div className="col-sm-1 float-right">
                <MdExpandLess
                  size={20}
                  fill="lightgrey"
                  data-toggle="tooltip"
                  title="Expand"
                  className="mr-2"
                  onClick={this.minimizeDiv}
                />
              </div>
            </div>
            <div>
              <div className="row">
                <div className="col-sm-12">
                  <div className="float-right" style={{ marginRight: "15px" }}>
                    <MdFlag
                      size={20}
                      fill="orange"
                      data-toggle="tooltip"
                      title="Incorrect suggestion"
                      className="mr-2 icon"
                      onClick={() =>
                        this.props.addToDictionary(
                          this.props.id,
                          this.props.title
                        )
                      }
                    />
                    <FcCheckmark
                      size={20}
                      data-toggle="tooltip"
                      title="Make Changes"
                      className="mr-2 icon"
                      onClick={this.changeContent}
                    />
                    <MdClear
                      size={20}
                      fill="red"
                      className="icon"
                      data-toggle="tooltip"
                      title="Discard suggestion"
                      onClick={() =>
                        this.props.deleteSuggestionCard(this.props.id)
                      }
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default SuggestionCard;
