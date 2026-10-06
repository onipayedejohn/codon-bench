/* Keeps Plotly charts in step with the light/dark theme. Colours come from the CSS tokens. */
(function () {
  var LIGHT_TO_TOKEN = {"#1baf7a": "--A", "#2a78d6": "--C", "#eda100": "--G", "#4a3aa7": "--T", "#e34948": "--U",
                        "#0e5e63": "--brand", "#3b4252": "--stop"};
  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function dark() { return document.documentElement.getAttribute("data-bs-theme") === "dark"; }
  function mapColor(c) {
    if (typeof c !== "string") return c;
    var t = LIGHT_TO_TOKEN[c.toLowerCase()];
    return t ? css(t) : c;
  }
  function apply(el) {
    if (!window.Plotly || !el || !el.data || !el.layout) return;
    if (!el._cbOrig) {                          // remember the colours the server sent
      el._cbOrig = el.data.map(function (t) {
        return {mc: t.marker ? JSON.parse(JSON.stringify(t.marker.color || null)) : null,
                lc: t.line ? t.line.color : null, mlc: t.marker && t.marker.line ? t.marker.line.color : null};
      });
    }
    var surface = css("--surface"), ink2 = css("--ink-2"), line = css("--line"), grid = css("--line-soft");
    var lay = {"paper_bgcolor": surface, "plot_bgcolor": surface, "font.color": ink2,
               "hoverlabel.bgcolor": surface, "hoverlabel.font.color": css("--ink"), "hoverlabel.bordercolor": line,
               "legend.font.color": ink2};
    Object.keys(el.layout).forEach(function (k) {
      if (/^[xy]axis\d*$/.test(k)) {
        lay[k + ".gridcolor"] = grid; lay[k + ".linecolor"] = line; lay[k + ".tickcolor"] = line;
        lay[k + ".tickfont.color"] = ink2; lay[k + ".title.font.color"] = ink2;
      }
    });
    (el.layout.annotations || []).forEach(function (a, i) { lay["annotations[" + i + "].font.color"] = ink2; });
    Plotly.relayout(el, lay);
    el.data.forEach(function (t, i) {
      var o = el._cbOrig[i] || {}, upd = {};
      if (o.mc !== null && o.mc !== undefined) {
        upd["marker.color"] = [dark() ? (Array.isArray(o.mc) ? o.mc.map(mapColor) : mapColor(o.mc)) : o.mc];
      }
      if (o.lc) upd["line.color"] = dark() ? mapColor(o.lc) : o.lc;
      if (o.mlc && /^#fff(fff)?$/i.test(o.mlc)) upd["marker.line.color"] = surface;
      if (Object.keys(upd).length) Plotly.restyle(el, upd, [i]);
    });
  }
  window.cbTheme = apply;
  function all() { document.querySelectorAll(".js-plotly-plot").forEach(apply); }
  new MutationObserver(all).observe(document.documentElement, {attributes: true, attributeFilter: ["data-bs-theme"]});
  document.addEventListener("shown.bs.tab", function () {
    document.querySelectorAll(".plot.js-plotly-plot").forEach(function (el) { if (el.offsetParent) Plotly.Plots.resize(el); });
  });
  document.addEventListener("click", function (e) {
    var l = e.target.closest(".navbar .nav-link"), c = document.querySelector(".navbar-collapse.show");
    if (l && c && window.bootstrap) bootstrap.Collapse.getOrCreateInstance(c).hide();
  });
})();
