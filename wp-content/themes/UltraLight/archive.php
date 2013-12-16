<?php
/*
Template Name: Archive
*/
?>

<?php get_header(); ?>

		<!-- Maincontent start -->
		<div id="maincontent">
			<h2>Archive</h2>

<?php while (have_posts()) : the_post(); ?>
			<div class="post">
				<h3><a href="<?php the_permalink() ?>" rel="bookmark" title="Permanent lenke til: <?php the_title_attribute(); ?>"><?php the_title(); ?></a></h3>
				<div class="date"><?php the_date(); ?> - <?php the_time(); ?></div>
				<?php the_content(''); ?>
			</div>
<?php endwhile; ?>

		</div>
		<!-- Maincontent end -->

<?php get_sidebar(); ?>

<?php get_footer(); ?>
