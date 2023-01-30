module Jekyll
  module Tags
    class IncludeChartTag < Liquid::Tag
      def initialize(tag_name, text, tokens)
        super
        params = text.split("!")
        @chart_name = params[0]
        @chart_x = params[1]
        @chart_y = params[2]
        @chart_config = params[3]
      end

      def render(context)
        r = Random.rand(10000)
        return p <<HERE
<div width="#{@chart_x}" height="#{@chart_y}">
  <canvas id="#{@chart_name}" border="2px" width="#{@chart_x}" height="#{@chart_y}"></canvas>
</div>

<script>
  const ctx_#{r} = document.getElementById('#{@chart_name}');
  new Chart(ctx_#{r}, #{@chart_config});
</script>
HERE
      end
    end
  end
end

Liquid::Template.register_tag("include_chart", Jekyll::Tags::IncludeChartTag)
