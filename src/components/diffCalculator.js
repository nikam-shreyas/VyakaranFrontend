var diff = require("fast-diff");

export default function diffCalculator(bad_sentence, good_sentence) {
  return diff(bad_sentence, good_sentence);
}
