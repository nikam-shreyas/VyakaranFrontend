import React, { Component } from "react";
import {
  MdClose,
  MdContentCopy,
  MdContentCut,
  MdContentPaste,
  MdSelectAll,
  MdFileDownload,
  MdPrint,
} from "react-icons/md";
import { FaKeyboard, FaLanguage } from "react-icons/fa";
import { HiOutlinePlusCircle } from "react-icons/hi";
class SideNav extends Component {
  state = { keyboardState: true };
  constructor(props) {
    super(props);
    this.toggleKeyboard = this.toggleKeyboard.bind(this);
    // this.uploadFile = this.uploadFile.bind(this);
    this.copyText = this.copyText.bind(this);
    this.cutText = this.cutText.bind(this);
    this.pasteText = this.pasteText.bind(this);
    this.printTextArea = this.printTextArea.bind(this);
  }
  reloadPage() {
    window.location.reload();
  }
  copyText() {
    var copyText = document.getElementById("content");
    copyText.select();
    copyText.setSelectionRange(0, 99999);
    document.execCommand("copy");
    this.closeSideNav();
  }
  cutText() {
    var cutText = document.getElementById("content");
    cutText.select();
    cutText.setSelectionRange(0, 99999);
    document.execCommand("cut");
    this.closeSideNav();
  }
  pasteText() {
    alert(
      "This action is not available from the menu. But, you can still use Ctrl+V"
    );
    this.closeSideNav();
  }
  toggleKeyboard() {
    if (this.state.keyboardState) {
      document.getElementById("keyboard_holder").style.display = "none";
      document.getElementById("content").rows = 20;
    } else {
      document.getElementById("keyboard_holder").style.display = "block";
      document.getElementById("content").rows = 8;
    }
    this.setState({ keyboardState: !this.state.keyboardState });
  }
  downloadFile() {
    var textToSave = document.getElementById("content").value;
    var textToSaveAsBlob = new Blob([textToSave], { type: "text/plain" });
    var textToSaveAsURL = window.URL.createObjectURL(textToSaveAsBlob);
    var fileNameToSaveAs = document.getElementById("doc_title").value;

    var downloadLink = document.createElement("a");
    downloadLink.download = fileNameToSaveAs;
    downloadLink.innerHTML = "Download File";
    downloadLink.href = textToSaveAsURL;

    downloadLink.style.display = "none";
    document.body.appendChild(downloadLink);

    downloadLink.click();
  }
  closeSideNav() {
    document.getElementById("mySidenav").style.width = "0px";
  }

  printTextArea() {
    var a = document.getElementById("doc_title").value;
    var b = document.getElementById("content").value;
    window.frames["print_frame"].document.title = document.title;
    window.frames["print_frame"].document.body.innerHTML =
      "<h2>" + a + "</h2>" + b;
    window.frames["print_frame"].window.focus();
    window.frames["print_frame"].window.print();
    // var WinPrint = window.open(
    //   "",
    //   "",
    //   "left=0,top=0,width=800,height=900,toolbar=0,scrollbars=0,status=0"
    // );
    // var prtContent = document.getElementById("doc_title");
    // WinPrint.document.write(
    //   "<script>function print(){document.execCommand('print')}</script><center onclick='print()'><a href='#'>Print</a><div id='printArea'</center><h3>" +
    //     prtContent.value +
    //     "</h3><br /><br />"
    // );

    // var prtContent = document.getElementById("content");
    // WinPrint.document.write(prtContent.value + "</div>");
    // WinPrint.focus();
    // WinPrint.close();
    // WinPrint.document.close();
  }
  //   uploadFile() {
  //     let uploadFileButton = document.createElement("input");
  //     uploadFileButton.type = "file";
  //     let tempDate = new Date().toISOString();
  //     uploadFileButton.id = tempDate;
  //     uploadFileButton.style.display = "none";
  //     document.body.appendChild(uploadFileButton);
  //     uploadFileButton.click();
  //     let fileToLoad = document.getElementById(tempDate).files[-1];

  //     let fileReader = new FileReader();
  //     fileReader.onload = function (fileLoadedEvent) {
  //       //   document.getElementById("doc_title").value = fileToLoad.name.slice(0, -4);
  //       var textFromFileLoaded = fileLoadedEvent.target.result;
  //       document.getElementById("content").value = textFromFileLoaded;
  //       fileReader.readAsText(fileToLoad, "UTF-8");
  //       this.closeSideNav();
  //     };
  //     // document.body.removeChild(uploadFileButton);
  //   }

  render() {
    return (
      <div id="mySidenav" className="sidenav">
        <ul>
          <li onClick={this.closeSideNav}>
            {" "}
            <MdClose fill="gray" style={{ marginTop: "-0.2vw" }} /> Close
          </li>
          <p>KEYBOARDS</p>
          <li onClick={this.toggleKeyboard}>
            <FaKeyboard size={18} fill="gray" style={{ marginTop: "-0.2vw" }} />{" "}
            Virtual Keyboard
          </li>
          <li>
            <FaLanguage size={18} fill="gray" style={{ marginTop: "-0.2vw" }} />{" "}
            Transliteration Keyboard
          </li>
          <p>DOCUMENT</p>
          <li onClick={this.reloadPage}>
            <HiOutlinePlusCircle style={{ marginTop: "-0.2vw" }} /> New Document
          </li>
          <li onClick={this.downloadFile}>
            <MdFileDownload fill="gray" style={{ marginTop: "-0.2vw" }} />{" "}
            Download
            <small className="float-right text-sm text-secondary mr-3">
              as .txt
            </small>
          </li>
          <li onClick={this.printTextArea}>
            <MdPrint fill="gray" style={{ marginTop: "-0.2vw" }} /> Print
            <small className="float-right text-sm text-secondary mr-3">
              Ctrl+P
            </small>
          </li>
          {/* <li onClick={this.uploadFile}>
            <MdFileUpload fill="gray" style={{ marginTop: "-0.2vw" }} /> Upload
            <small className="float-right text-sm text-secondary mr-3">
              as .txt
            </small>
          </li> */}
          <p>EDIT</p>

          <li onClick={this.cutText}>
            <MdContentCut fill="gray" style={{ marginTop: "-0.2vw" }} /> Cut
            <small className="float-right text-sm text-secondary mr-3">
              Ctrl+X
            </small>
          </li>
          <li onClick={this.copyText}>
            <MdContentCopy fill="gray" style={{ marginTop: "-0.2vw" }} /> Copy
            <small className="float-right text-sm text-secondary mr-3">
              Ctrl+C
            </small>
          </li>
          <li onClick={this.pasteText}>
            <MdContentPaste fill="gray" style={{ marginTop: "-0.2vw" }} /> Paste
            <small className="float-right text-sm text-secondary mr-3">
              Ctrl+V
            </small>
          </li>
          <li>
            <MdSelectAll fill="gray" style={{ marginTop: "-0.2vw" }} /> Select
            all
            <small className="float-right text-sm text-secondary mr-3">
              Ctrl+A
            </small>
          </li>
        </ul>
      </div>
    );
  }
}

export default SideNav;
