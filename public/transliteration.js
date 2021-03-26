eval(
  (function (p, a, c, k, e, d) {
    e = function (c) {
      return (
        (c < a ? "" : e(parseInt(c / a))) +
        ((c = c % a) > 35 ? String.fromCharCode(c + 29) : c.toString(36))
      );
    };
    if (!"".replace(/^/, String)) {
      while (c--) {
        d[e(c)] = k[c] || e(c);
      }
      k = [
        function (e) {
          return d[e];
        },
      ];
      e = function () {
        return "\\w+";
      };
      c = 1;
    }
    while (c--) {
      if (k[c]) {
        p = p.replace(new RegExp("\\b" + e(c) + "\\b", "g"), k[c]);
      }
    }
    return p;
  })(
    "4.1x(\"<y G=\\\"12:10;11:10;\\\" 6='5-1b' ><y G='1a:15;11:13;12:13;' 6='e-16'>17 1d:</y> <a 6='5-e' b=\\\"5#a\\\" m=\\\"g:(p(){f(J 9 != 'V') {K='#5#'; 7 { X(); } 8(E) {}; 9.R(T); Q=0; 7 {Y();} 8(S) {}; P();F; } Z { t=N M().L();4.v.r(4.u('h')).j='A://U.5.B/O/?I=D&l=0&k=1&n=&c=&q='+t;}})();\\\">5</a> <a 6='C-e' b=\\\"C#a\\\" m=\\\"g:(p(){f(J 9 != 'V'){K='#C#'; 7 { X();} 8(E) {}; 9.R(T); Q=0; 7 { Y();} 8(S) {}; 	P();F; };t=N M().L();4.v.r(4.u('h')).j='A://U.5.B/O/?I=D&l=0&k=2&n=&c=&q='+t;})();\\\">C</a> <a 6='H-e' b=\\\"H#a\\\" m=\\\"g:(p(){f(J 9 != 'V'){K='#H#';7 { X();} 8(E) {}; 9.R(T); Q=0; 7 { Y(); } 8(S) {};	P();F; };t=N M().L();4.v.r(4.u('h')).j='A://U.5.B/O/?I=D&l=0&k=3&n=&c=&q='+t;})();\\\">1c</a> <a 6='18-e' m=\\\"g:(W=19.W||p(l){1f t=W,d=4,o=d.v,c='u',a='r',w='1e',i=d[c]('1A'),s=i.G,x=o[a](d[c]('h'));f(o){f(!t.l){t.l=x.b='1D';o[a](i).b='14';i.1E='1F 1G';s.1H='z-1C:1B;1z-1I:1y;1o:#1w;1h:0';s.1i=d.D?'1j':'1k';s.15=((o[w]-i[w])/2)+'1l';x.j='A://14.1m.B/1g/1n/1p/1q.1r?l='+l}}Z 1s(t,1t)})('1u')\\\">1v</a></y>\");",
    62,
    107,
    "||||document|gamabhana|class|try|catch|gamabhanaExt||id|||opt|if|javascript|script||src|||href|||function||appendChild|||createElement|body|||div||https|com|fontfreedom|all|x1|return|style|inscript|mode|typeof|__predefKb|getTime|Date|new|gamabhanaWidget|attemptRun|__predefLangIndex|changeLayout|x2|null|www|undefined|t13nb|handlePreDefKB|handlePreDefLang|else|2px|padding|margin|0px|t13n|left|label|type|google|window|float|widget|Inscript|with|clientWidth|var|svn|top|position|absolute|fixed|px|googlecode|trunk|background|blet|rt13n|js|setTimeout|500|mr|Google|FFF1A8|write|18px|font|span|99|index|t13ns|innerHTML|Loading|Transliteration|cssText|size".split(
      "|"
    ),
    0,
    {}
  )
);
