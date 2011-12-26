<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" <?php language_attributes(); ?>>
<head profile="http://gmpg.org/xfn/11">
<meta http-equiv="Content-Type" content="<?php bloginfo('html_type'); ?>; charset=<?php bloginfo('charset'); ?>" />
<title><?php display_title($s); ?></title>
<link type="text/css" rel="stylesheet" href="<?php echo get_stylesheet_uri(); ?>" media="screen" />
<?php if (strpos ($_SERVER['HTTP_USER_AGENT'], 'Presto')) : ?>
<style type="text/css">
#gray-shading { margin-top:-16px; }
ul#top-nav ul li { margin-left:-80px; }
ul#top-nav ul li a { margin-left:30px; }
ul#top-nav ul ul li { margin-left:0; }
</style>
<?php endif; ?>
<?php if (strpos ($_SERVER['HTTP_USER_AGENT'], 'MSIE 7')) : ?>
<link type="text/css" rel="stylesheet" href="<?php echo get_template_directory_uri(); ?>/lib/css/css-for-stupid-browsers-named-ie.css" class="clearfix" />
<?php endif; ?>
<link rel="pingback" href="<?php bloginfo('pingback_url'); ?>" />
<link rel="shortcut icon" href="<?php echo get_template_directory_uri(); ?>/lib/images/favicon.ico" />
<?php
wp_register_script('livequery',get_template_directory_uri().'/lib/javascript/jquery-livequery.js',array('jquery'));
wp_register_script('website',get_template_directory_uri().'/lib/javascript/website.js',array('jquery','livequery'));
if(is_singular()):
wp_enqueue_script( 'comment-reply' );
else:
wp_enqueue_script('jquery');
wp_enqueue_script('livequery');
wp_enqueue_script('website');
endif;
?>
<?php wp_head(); ?>
</head>
<body <?php body_class(); ?>>
		
		<div id="horz-wrapper">
			
			<div id="title-slogan">
                <?php get_search_form(); ?>
				<h1><a href="<?php echo home_url(); ?>/"><?php bloginfo('name'); ?></a></h1>
				<p class="slogan"><?php bloginfo('description'); ?></p>
			</div>
			
			<?php mytheme_nav(); ?>
			<br class="clear"/>