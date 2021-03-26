import React, { Component } from "react";
import { MdArrowForward, MdClear } from "react-icons/md";
import { FcCheckmark } from "react-icons/fc";
class SuggestionCard extends Component {
  state = {};
  constructor(props) {
    super(props);
    this.changeContent = this.changeContent.bind(this);
  }
  changeContent() {
    this.props.replaceContent(
      this.props.title,
      this.props.correction,
      this.props.index
    );
    // document.getElementById("content").value = temp;
    this.props.deleteSuggestionCard(this.props.id);
  }
  render() {
    return (
      <div className="cardx" id={this.props.id}>
        <div className="cardx-body">
          <div className="row">
            <div className="col-sm-8 center">
              <span
                className="circle pulse rose ml-2"
                style={{ marginTop: "10px" }}
              ></span>
              <span className="ml-3">
                {this.state.info}
                <small>{this.props.title}</small>
                {"   "}
                <MdArrowForward fill="lightgray" />
                {"   "}
                <small>{this.props.correction}</small>
              </span>
            </div>
            <div className="col-sm-4">
              <div className="float-right expansion_icon">
                <FcCheckmark
                  size={20}
                  data-toggle="tooltip"
                  title="Make Changes"
                  className="mr-2"
                  onClick={this.changeContent}
                />
                <MdClear
                  size={20}
                  fill="red"
                  data-toggle="tooltip"
                  title="Discard suggestion"
                  onClick={() => this.props.deleteSuggestionCard(this.props.id)}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default SuggestionCard;
