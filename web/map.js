/* Draws the model's zones as SVG panels and recolours them. Generic: any panels, any zones.
   Each zone has an opaque base (looks like land) and a coloured overlay whose fill-opacity
   follows the probability; CSS fades between updates. Exposes the global AccentMap. */
var AccentMap = (function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(e);
    return e;
  }

  function hatch(defs, id, size) {
    var p = el("pattern", { id: id, patternUnits: "userSpaceOnUse", width: size, height: size,
      patternTransform: "rotate(45)" }, defs);
    el("rect", { class: "hatch-bg", width: size, height: size }, p);
    el("line", { class: "hatch-line", x1: 0, y1: 0, x2: 0, y2: size, "stroke-width": size / 3 }, p);
  }

  // opts.onHover(info, pointerEvent) with info = {zone} | {label, text} | null
  function create(container, panels, geojson, opts) {
    var overlays = {};   // zone -> overlay paths (a zone can appear in several panels)
    var hover = opts.onHover || function () {};

    panels.forEach(function (panel, index) {
      var proj = AccentLogic.projection(panel.bbox);
      var figure = document.createElement("figure");
      figure.className = "panel " + (index === 0 ? "main" : "inset");
      var svg = el("svg", { viewBox: "0 0 " + proj.width.toFixed(2) + " " + proj.height.toFixed(2),
        role: "img", "aria-label": "Map: " + (panel.title || panel.id) });
      var defs = el("defs", {}, svg);
      var hatchId = "hatch-" + panel.id;
      hatch(defs, hatchId, proj.width / 110);
      el("rect", { class: "sea", width: proj.width, height: proj.height }, svg);
      var layers = { land: el("g", {}, svg), uncovered: el("g", {}, svg), zone: el("g", {}, svg) };

      geojson.features.forEach(function (f) {
        var props = f.properties;
        if (props.panel !== panel.id) return;
        var d = AccentLogic.geometryPath(f.geometry, proj);
        if (props.kind === "land") {
          el("path", { d: d, class: "land" }, layers.land);
        } else if (props.kind === "uncovered") {
          var u = el("path", { d: d, class: "uncovered", fill: "url(#" + hatchId + ")", "fill-rule": "evenodd" },
            layers.uncovered);
          var info = { label: props.label, text: "Spanish is spoken here, but this model has no class for it." };
          u.addEventListener("pointermove", function (e) { hover(info, e); });
          u.addEventListener("pointerleave", function (e) { hover(null, e); });
        } else if (props.kind === "zone") {
          el("path", { d: d, class: "zone-base", "fill-rule": "evenodd" }, layers.zone);
          var o = el("path", { d: d, class: "zone-fill", "fill-rule": "evenodd", "data-zone": props.zone },
            layers.zone);
          o.style.fillOpacity = 0;
          (overlays[props.zone] = overlays[props.zone] || []).push(o);
          o.addEventListener("pointermove", function (e) { hover({ zone: props.zone }, e); });
          o.addEventListener("pointerleave", function (e) { hover(null, e); });
        }
      });

      figure.appendChild(svg);
      if (panel.title) {
        var caption = document.createElement("figcaption");
        caption.textContent = panel.title;
        figure.appendChild(caption);
      }
      container.appendChild(figure);
    });

    return {
      // zoneProbs: {zone: p}; null clears the colours.
      update: function (zoneProbs, status) {
        Object.keys(overlays).forEach(function (zone) {
          var opacity = zoneProbs ? AccentLogic.zoneOpacity(zoneProbs[zone] || 0, status) : 0;
          overlays[zone].forEach(function (path) { path.style.fillOpacity = opacity; });
        });
      },
      highlight: function (zone) {
        Object.keys(overlays).forEach(function (z) {
          overlays[z].forEach(function (path) { path.classList.toggle("hl", z === zone); });
        });
      }
    };
  }

  return { create: create };
})();
