<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" 
  "http://www.w3.org/TR/html4/loose.dtd">
<html lang="en">



<head>
<title><?php bloginfo('name'); ?></title>
<meta http-equiv="Content-Type" content="<?php bloginfo('html_type'); ?>; charset=<?php bloginfo('charset'); ?>">


<meta name="generator" content="WordPress <?php bloginfo('version'); ?>">



<link rel="stylesheet" href="<?php bloginfo('stylesheet_url'); ?>" type="text/css" media="screen">

<link rel="alternate" type="application/rss+xml" title="<?php bloginfo('name'); ?> RSS Feed" href="<?php bloginfo('rss2_url'); ?>">

<link rel="pingback" href="<?php bloginfo('pingback_url'); ?>">

</head>

<body>
	<div id="container">
		<div id="logo"><?php bloginfo('title'); ?></div>
		<ul id="nav">
			<li<?php if ( is_home() or is_archive() or is_single() or is_paged() or is_search() or (function_exists('is_tag') and is_tag()) ) { echo ' class="current_page_item"'; } ?>><a href="<?php bloginfo('url'); ?>">Home</a></li><?php wp_list_pages('sort_column=id&depth=1&title_li=&exclude=9,10'); ?>
		</ul>
		<div class="clear"></div>
	</div>
	<div class="spacer"></div>
	<div class="content">