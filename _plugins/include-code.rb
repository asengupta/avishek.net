require 'net/http'
require 'uri'

module Jekyll
  module Tags
    class IncludeCodeTag < Liquid::Tag
      def initialize(tag_name, text, tokens)
        super
        params = text.split("!")
        @chart_name = URI.parse(params[0])
        @from = params[1].to_i - 1
        @to = params[2].to_i - 1
      end

      def render(context)
        response = Net::HTTP.get_response(@chart_name)
        response.body.lines[Range.new(@from, @to)]
      end
    end
  end
end

Liquid::Template.register_tag("include_code", Jekyll::Tags::IncludeCodeTag)
