<!-- UltraLight - A WordPress theme by Fredrik Sørlie -->
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" dir="ltr">

<head>

<title><?php bloginfo('name'); ?> <?php bloginfo('description'); ?> <?php wp_title(); ?></title>

<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<meta name="generator" content="WordPress <?php bloginfo('version'); ?>" />
<meta name="Keywords" content="ultralight, wordpress, theme" />
<meta name="author" content="Fredrik Sørlie - Design and Communication" />
<meta name="copyright" content="Copyright <?php the_date('Y'); ?> <?php bloginfo('name'); ?>" />
<meta name="robots" content="all" />

<link rel="credits" href="http://fredriksoerlie.com" type="rel" />
<link rel="shortcut icon" href="images/favicon.ico" />
<link rel="stylesheet" href="<?php bloginfo('stylesheet_url'); ?>" type="text/css" media="screen" />

<!--[if lt IE 7.]>
<script defer type="text/javascript" src="pngfix.js"></script>
<![endif]-->

<!-- WP_head start -->
<?php wp_head(); ?>
<!-- WP_head end -->

</head>

<body>

<div id="page">

	<!-- Header start -->
	<div id="header">
		<h1 id="logo"><a href="<?php echo get_option('home'); ?>/"><img src="<?php bloginfo('template_directory'); ?>/images/logo.png" alt="<?php bloginfo('name'); ?>" /></a></h1>
	</div>
	<!-- Header end -->

	<!-- Menu start -->
	<ul id="menu">
		<li<?php if(is_home()){echo ' class="current_page_item"';}?>><a href="<?php bloginfo('siteurl'); ?>" title="Home">Home</a></li>
		<?php wp_list_pages('title_li=&depth=1&sort_column=menu_order');?>
		<li class="right"><a href="/feed" class="rss">RSS Feed</a></li>
	</ul>
	<!-- Menu end -->

	<!-- Content start -->
	<div id="content">
