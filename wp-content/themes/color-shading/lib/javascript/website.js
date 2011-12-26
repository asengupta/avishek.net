jQuery(function ($) {
	$('#accordion > li > p').each(function (i,dom) {
		$(dom).css('height', $(dom).outerHeight());
	}).hide();
	$('#widgetacc > li > p').each(function (i,dom) {
		$(dom).css('height', $(dom).outerHeight());
	}).hide();
	$('#accordion > li > a').hover(function () {
		$(this).next('p:not(:animated)').slideDown(500);
	}, function () {
	});
	$('#widgetacc > li > a').hover(function () {
		$(this).next('p:not(:animated)').slideDown(500);
	}, function () {
	});
	$('#accordion > li').hover(function () {
		$(this).addClass('hover');
	}, function () {
		$(this).removeClass('hover');
		$(this).children('p').slideUp(500);
	});
	$('#widgetacc > li').hover(function () {
		$(this).addClass('hover');
	}, function () {
		$(this).removeClass('hover');
		$(this).children('p').slideUp(500);
	});
	$("table tr").mouseover(function() {$(this).addClass("over");}).mouseout(function() {$(this).removeClass("over");});
	
	$("table").each(function (i, dom) {
		$(dom).find("tr:even").addClass("alt");
		$(dom).find("tr:last").addClass("last");
	});
	$("input").livequery(function () {
		$(this).focus(function() {
			// only select if the text has not changed
			$(this).addClass('focus');
			if(this.value === this.defaultValue)	{
				this.select();
			}
		});
	});
	$("input, textarea").livequery(function () {
		$(this).blur(function() {
			// only select if the text has not changed
			$(this).removeClass('focus');
		});
	});
	$("input:submit, button").livequery(function() {
		$(this).addClass('button');
	});
	$("input:checkbox").livequery(function() {
		$(this).addClass('checkbox');
	});
	$("input:radio").livequery(function() {
		$(this).addClass('radio');
	});
	$("input, button").livequery(function() {
		$(this).hover(function () {
			$(this).toggleClass('hover');
		},
		function () {
			$(this).toggleClass('hover');
		});
	});
});