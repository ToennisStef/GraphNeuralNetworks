function drawGraph(nodes, links) {
  const svg = d3.select("svg");
  const width = window.innerWidth;
  const height = window.innerHeight;

  // Define a color scale for the node values
  // const colorScale = d3.scaleOrdinal()
  //   .domain([1, 2])
  //   .range(["blue", "orange"]); // Blue for value 1, Orange for value 2
 


  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2));

  // Create an expanded links array
  const processedLinks = [];
  links.forEach(link => {
    if (link.value[0] === 2) {
      // Push two copies with different offsets
      processedLinks.push({ ...link, offset: 3 });
      processedLinks.push({ ...link, offset: -3 });
    } else {
      // Normal link, zero offset
      processedLinks.push({ ...link, offset: 0 });
    }
  });

  const link = svg.append("g")
    .selectAll("line")
    .data(processedLinks)
    .join("line")
    .attr("stroke", "#aaa")
    .attr("stroke-width", 5);
    
  const node = svg.append("g")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .call(drag(simulation));

  node.append("circle")
    .attr("r", 20)
    .attr("fill", d => {
      if (d.value[0] === 1) return "blue";
      if (d.value[0] === 2) return "orange";
      if (d.value[0] === 3) return "green";
      if (d.value[0] === 4) return "red";
      if (d.value[0] === 5) return "purple";
      if (d.value[0] === 6) return "pink";
      return "gray";
    });


  node.append("text")
    .text(d => d.id)
    .attr("text-anchor", "middle")
    .attr("dy", "0.35em");

  function offsetLineCoords(d) {
    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const angle = Math.atan2(dy, dx);
    const offsetX = d.offset * Math.sin(angle);
    const offsetY = -d.offset * Math.cos(angle);
    return {
      x1: d.source.x + offsetX,
      y1: d.source.y + offsetY,
      x2: d.target.x + offsetX,
      y2: d.target.y + offsetY
    };
  }

  simulation.on("tick", () => {
    link
      .attr("x1", d => offsetLineCoords(d).x1)
      .attr("y1", d => offsetLineCoords(d).y1)
      .attr("x2", d => offsetLineCoords(d).x2)
      .attr("y2", d => offsetLineCoords(d).y2);
      // .attr("x1", d => d.source.x)
      // .attr("y1", d => d.source.y)
      // .attr("x2", d => d.target.x)
      // .attr("y2", d => d.target.y);

    node
      .attr("transform", d => `translate(${d.x},${d.y})`);
  });

  function drag(simulation) {
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // node.append("title")
    //   .text(d => `Value: ${d.value}`);
    
    return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
  }

  // Create a tooltip element
  const tooltip = d3.select("body").append("div")
  .attr("class", "tooltip")
  .style("position", "absolute")
  .style("background-color", "white")
  .style("border", "1px solid #ccc")
  .style("padding", "5px")
  .style("border-radius", "5px")
  .style("display", "none");

  // Add mouseover and mouseout events for the tooltip
  node.on("mouseover", (event, d) => {
      tooltip.style("display", "block")
        .html(`Atom: ${d.atom}`)
        .style("left", `${event.pageX + 10}px`)
        .style("top", `${event.pageY + 10}px`);
    })
    .on("mouseout", () => {
      tooltip.style("display", "none");
    });

  // Add mouseover and mouseout events for the links
  link.on("mouseover", (event, d) => {
    tooltip.style("display", "block")
      .html(`Bond: ${d.bond}`)
      .style("left", `${event.pageX + 10}px`)
      .style("top", `${event.pageY + 10}px`);
    })
    .on("mouseout", () => {
    tooltip.style("display", "none");
    });

}