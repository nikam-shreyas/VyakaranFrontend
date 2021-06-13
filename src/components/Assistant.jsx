import React, { Component } from "react";
import { MdClose } from "react-icons/md";
class Assistant extends Component {
  state = {
    score: 0.0,
    accuracyWidth: 0,
    mistakesWidth: 0,
    remark: "Please start typing.",
    totalLength: 0,
    mistakes: 0,
    remark_class: "remark_0",
  };
  constructor(props) {
    super(props);
  }
  componentWillReceiveProps(props) {
    if (props.totalLength > 0) {
      let correct = props.totalLength - props.mistakesLength;
      let remark = "";
      let remark_class = "";
      let temp = correct / props.totalLength;
      let sw = Math.floor((correct * 10) / props.totalLength);
      switch (sw) {
        case 0:
        case 1:
          remark = "Poor";
          remark_class = "remark_1";
          break;
        case 2:
        case 3:
          remark = "Needs Improvement";
          remark_class = "remark_2";
          break;
        case 4:
        case 5:
          remark = "Average";
          remark_class = "remark_3";
        case 6:
        case 7:
          remark = "Above Average";
          remark_class = "remark_4";
          break;
        case 8:
        case 9:
          remark = "Excellent";
          remark_class = "remark_5";
          break;
      }

      this.setState({
        score: (temp * 100).toFixed(2),
        accuracyWidth: (correct * 100) / props.totalLength,
        mistakesWidth: (props.mistakesLength * 100) / props.totalLength,
        remark: remark,
        remark_class: remark_class,
        mistakes: props.mistakes,
      });
    }
  }
  componentDidMount() {}
  closeAssistant() {
    document.getElementById("myAssistant").style.width = "0px";
  }
  render() {
    return (
      <div id="myAssistant" className="assistant">
        <ul>
          <li onClick={this.closeAssistant}>
            Hide Assistant{" "}
            <MdClose fill="gray" style={{ marginTop: "-0.2vw" }} />
          </li>
          <hr />
          <li>
            <h6>
              <b>Overview</b>
            </h6>
          </li>
          <li>
            <b>Score:</b>
            <br />
            <small>{this.state.score} %</small>
          </li>
          <li>
            <b>Remark:</b>
            <br />
            <small className={this.state.remark_class}>
              {this.state.remark}
            </small>
          </li>
          <hr />
          <li>
            <h6>
              <b>Performance</b>
            </h6>
          </li>
          <li>
            <b>Accuracy:</b>

            <div
              class="progress"
              style={{ height: "3px", marginRight: "15px" }}
            >
              <div
                class="progress-bar progress-bar-striped bg-success progress-bar-animated"
                role="progressbar"
                aria-valuenow="75"
                aria-valuemin="0"
                aria-valuemax="100"
                style={{
                  width: this.state.accuracyWidth + "%",
                  height: "3px",
                  marginRight: "15px",
                }}
              ></div>
            </div>
          </li>
          <li>
            <b>Mistakes:</b>

            <div
              class="progress"
              style={{ height: "3px", marginRight: "15px" }}
            >
              <div
                class="progress-bar progress-bar-striped bg-danger progress-bar-animated"
                role="progressbar"
                aria-valuenow="75"
                aria-valuemin="0"
                aria-valuemax="100"
                style={{
                  width: this.state.mistakesWidth + "%",
                  height: "3px",
                  marginRight: "15px",
                }}
              ></div>
            </div>
          </li>
          <li>
            <b>Blunders:</b>
            <br />

            <small className="error_title">
              <b>{this.state.mistakes}</b>
            </small>
          </li>
        </ul>
      </div>
    );
  }
}

export default Assistant;
